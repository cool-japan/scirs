//! Box-Cox transformations
//!
//! This module implements the Box-Cox family of power transformations,
//! commonly used in statistics for data normalization and variance stabilization.
//!
//! ## Mathematical Theory
//!
//! The Box-Cox transformation is defined as:
//! ```text
//! y(λ) = {
//!   (x^λ - 1) / λ    if λ ≠ 0
//!   log(x)           if λ = 0
//! }
//! ```
//!
//! ### Properties
//! 1. **Continuity**: The transformation is continuous at λ = 0
//! 2. **Monotonicity**: For x > 0, y(λ) is strictly increasing in x
//! 3. **Variance Stabilization**: Can help achieve homoscedasticity
//! 4. **Normality**: Often helps achieve approximate normality
//!
//! ### Applications
//! - Linear regression with non-normal residuals
//! - Time series analysis
//! - ANOVA with heteroscedastic errors
//! - Data preprocessing for machine learning

#![allow(dead_code)]

use crate::{SpecialError, SpecialResult};
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::numeric::{Float, FromPrimitive};
use scirs2_core::validation::{check_finite, check_positive};
use std::fmt::{Debug, Display};

/// Box-Cox transformation
///
/// Applies the Box-Cox power transformation:
/// - For λ ≠ 0: (x^λ - 1) / λ
/// - For λ = 0: log(x)
///
/// # Arguments
/// * `x` - Input value (must be positive)
/// * `lmbda` - Transformation parameter λ
///
/// # Returns
/// Transformed value
///
/// # Examples
/// ```
/// use scirs2_special::boxcox;
/// use std::f64::consts::E;
///
/// // λ = 0 case (logarithmic)
/// let result = boxcox(E, 0.0).unwrap();
/// assert!((result - 1.0f64).abs() < 1e-10);
///
/// // λ = 1 case (identity minus 1)
/// let result = boxcox(5.0, 1.0).unwrap();
/// assert!((result - 4.0f64).abs() < 1e-10);
///
/// // λ = 0.5 case (square root transformation)
/// let result = boxcox(4.0, 0.5).unwrap();
/// assert!((result - 2.0f64).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn boxcox<T>(x: T, lmbda: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Display + Copy + Debug,
{
    check_positive(x, "x")?;
    check_finite(x, "x value")?;
    check_finite(lmbda, "lmbda value")?;

    let _zero = T::from_f64(0.0).unwrap();
    let one = T::one();

    if lmbda.abs() < T::from_f64(1e-10).unwrap() {
        // λ ≈ 0: use logarithmic transformation
        Ok(x.ln())
    } else {
        // λ ≠ 0: use power transformation
        Ok((x.powf(lmbda) - one) / lmbda)
    }
}

/// Box-Cox transformation of 1 + x
///
/// Applies the Box-Cox transformation to (1 + x):
/// - For λ ≠ 0: ((1+x)^λ - 1) / λ
/// - For λ = 0: log(1+x)
///
/// This variant is useful when x can be close to zero.
///
/// # Arguments
/// * `x` - Input value (must be > -1)
/// * `lmbda` - Transformation parameter λ
///
/// # Examples
/// ```
/// use scirs2_special::boxcox1p;
///
/// // λ = 0 case
/// let result = boxcox1p(1.718, 0.0).unwrap();
/// assert!((result - (1.0 + 1.718_f64).ln()).abs() < 1e-10);
///
/// // Small x values
/// let result: f64 = boxcox1p(0.01, 0.5).unwrap();
/// assert!(result.is_finite());
/// ```
#[allow(dead_code)]
pub fn boxcox1p<T>(x: T, lmbda: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Display + Copy + Debug,
{
    check_finite(x, "x value")?;
    check_finite(lmbda, "lmbda value")?;

    let neg_one = T::from_f64(-1.0).unwrap();
    if x <= neg_one {
        return Err(SpecialError::DomainError(
            "x must be greater than -1".to_string(),
        ));
    }

    let _zero = T::from_f64(0.0).unwrap();
    let one = T::one();
    let one_plus_x = one + x;

    if lmbda.abs() < T::from_f64(1e-10).unwrap() {
        // λ ≈ 0: use log(1+x)
        Ok(one_plus_x.ln())
    } else {
        // λ ≠ 0: use power transformation
        Ok((one_plus_x.powf(lmbda) - one) / lmbda)
    }
}

/// Inverse Box-Cox transformation
///
/// Computes the inverse of the Box-Cox transformation:
/// - For λ ≠ 0: (λ*y + 1)^(1/λ)
/// - For λ = 0: exp(y)
///
/// # Arguments
/// * `y` - Transformed value
/// * `lmbda` - Transformation parameter λ
///
/// # Examples
/// ```
/// use scirs2_special::{boxcox, inv_boxcox};
///
/// let x = 5.0f64;
/// let lmbda = 0.3f64;
/// let y = boxcox(x, lmbda).unwrap();
/// let x_recovered = inv_boxcox(y, lmbda).unwrap();
/// assert!((x - x_recovered).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn inv_boxcox<T>(y: T, lmbda: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Display + Copy + Debug,
{
    check_finite(y, "y value")?;
    check_finite(lmbda, "lmbda value")?;

    let zero = T::from_f64(0.0).unwrap();
    let one = T::one();

    if lmbda.abs() < T::from_f64(1e-10).unwrap() {
        // λ ≈ 0: inverse of log is exp
        Ok(y.exp())
    } else {
        // λ ≠ 0: (λ*y + 1)^(1/λ)
        let lambda_y_plus_1 = lmbda * y + one;

        if lambda_y_plus_1 <= zero {
            return Err(SpecialError::DomainError(
                "Invalid argument for inverse transformation".to_string(),
            ));
        }

        Ok(lambda_y_plus_1.powf(one / lmbda))
    }
}

/// Inverse Box-Cox transformation of 1 + x form
///
/// Computes the inverse of boxcox1p:
/// - For λ ≠ 0: (λ*y + 1)^(1/λ) - 1
/// - For λ = 0: exp(y) - 1
///
/// # Arguments
/// * `y` - Transformed value
/// * `lmbda` - Transformation parameter λ
///
/// # Examples
/// ```
/// use scirs2_special::{boxcox1p, inv_boxcox1p};
///
/// let x = 0.5f64;
/// let lmbda = -0.2f64;
/// let y = boxcox1p(x, lmbda).unwrap();
/// let x_recovered = inv_boxcox1p(y, lmbda).unwrap();
/// assert!((x - x_recovered).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn inv_boxcox1p<T>(y: T, lmbda: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Display + Copy + Debug,
{
    check_finite(y, "y value")?;
    check_finite(lmbda, "lmbda value")?;

    let zero = T::from_f64(0.0).unwrap();
    let one = T::one();

    if lmbda.abs() < T::from_f64(1e-10).unwrap() {
        // λ ≈ 0: inverse of log(1+x) is exp(y) - 1
        Ok(y.exp() - one)
    } else {
        // λ ≠ 0: (λ*y + 1)^(1/λ) - 1
        let lambda_y_plus_1 = lmbda * y + one;

        if lambda_y_plus_1 <= zero {
            return Err(SpecialError::DomainError(
                "Invalid argument for inverse transformation".to_string(),
            ));
        }

        Ok(lambda_y_plus_1.powf(one / lmbda) - one)
    }
}

/// Box-Cox transformation for arrays
///
/// Applies Box-Cox transformation element-wise to an array.
///
/// # Arguments
/// * `x` - Input array (all elements must be positive)
/// * `lmbda` - Transformation parameter λ
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_special::boxcox_array;
///
/// let x = array![1.0, 2.0, 4.0, 8.0];
/// let result = boxcox_array(&x.view(), 0.0).unwrap();
/// // Should be approximately [0, ln(2), ln(4), ln(8)]
/// ```
#[allow(dead_code)]
pub fn boxcox_array<T>(x: &ArrayView1<T>, lmbda: T) -> SpecialResult<Array1<T>>
where
    T: Float + FromPrimitive + Display + Copy + Debug,
{
    let mut result = Array1::zeros(x.len());

    for (i, &val) in x.iter().enumerate() {
        result[i] = boxcox(val, lmbda)?;
    }

    Ok(result)
}

/// Box-Cox1p transformation for arrays
#[allow(dead_code)]
pub fn boxcox1p_array<T>(x: &ArrayView1<T>, lmbda: T) -> SpecialResult<Array1<T>>
where
    T: Float + FromPrimitive + Display + Copy + Debug,
{
    let mut result = Array1::zeros(x.len());

    for (i, &val) in x.iter().enumerate() {
        result[i] = boxcox1p(val, lmbda)?;
    }

    Ok(result)
}

/// Inverse Box-Cox transformation for arrays
#[allow(dead_code)]
pub fn inv_boxcox_array<T>(y: &ArrayView1<T>, lmbda: T) -> SpecialResult<Array1<T>>
where
    T: Float + FromPrimitive + Display + Copy + Debug,
{
    let mut result = Array1::zeros(y.len());

    for (i, &val) in y.iter().enumerate() {
        result[i] = inv_boxcox(val, lmbda)?;
    }

    Ok(result)
}

/// Inverse Box-Cox1p transformation for arrays
#[allow(dead_code)]
pub fn inv_boxcox1p_array<T>(y: &ArrayView1<T>, lmbda: T) -> SpecialResult<Array1<T>>
where
    T: Float + FromPrimitive + Display + Copy + Debug,
{
    let mut result = Array1::zeros(y.len());

    for (i, &val) in y.iter().enumerate() {
        result[i] = inv_boxcox1p(val, lmbda)?;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_boxcox_basic() {
        // Test λ = 0 (logarithmic case)
        let result = boxcox(std::f64::consts::E, 0.0).unwrap();
        assert_relative_eq!(result, 1.0, epsilon = 1e-10);

        // Test λ = 1 (linear case minus 1)
        let result = boxcox(5.0, 1.0).unwrap();
        assert_relative_eq!(result, 4.0, epsilon = 1e-10);

        // Test λ = 0.5 (square root case)
        let result = boxcox(4.0, 0.5).unwrap();
        assert_relative_eq!(result, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_boxcox1p_basic() {
        // Test λ = 0
        let result = boxcox1p(1.718, 0.0).unwrap();
        let expected = (1.0 + 1.718_f64).ln();
        assert_relative_eq!(result, expected, epsilon = 1e-10);

        // Test small values
        let result = boxcox1p(0.01, 0.5).unwrap();
        assert!(result.is_finite());
    }

    #[test]
    fn test_inverse_properties() {
        let test_values = [0.1, 1.0, 2.0, 5.0, 10.0];
        let lambdas = [-0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0];

        for &x in &test_values {
            for &lmbda in &lambdas {
                // Test boxcox and inv_boxcox
                let y = boxcox(x, lmbda).unwrap();
                let x_recovered = inv_boxcox(y, lmbda).unwrap();
                assert_relative_eq!(x, x_recovered, epsilon = 1e-10);

                // Test boxcox1p and inv_boxcox1p
                let y1p = boxcox1p(x, lmbda).unwrap();
                let x1p_recovered = inv_boxcox1p(y1p, lmbda).unwrap();
                assert_relative_eq!(x, x1p_recovered, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_array_operations() {
        let x = array![1.0, 2.0, 4.0, 8.0];
        let lmbda = 0.5;

        let result = boxcox_array(&x.view(), lmbda).unwrap();
        assert_eq!(result.len(), 4);

        // Test round-trip
        let recovered = inv_boxcox_array(&result.view(), lmbda).unwrap();
        for i in 0..4 {
            assert_relative_eq!(x[i], recovered[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_error_conditions() {
        // Negative input for boxcox
        assert!(boxcox(-1.0, 0.5).is_err());

        // Input <= -1 for boxcox1p
        assert!(boxcox1p(-1.0, 0.5).is_err());
        assert!(boxcox1p(-1.1, 0.5).is_err());
    }

    #[test]
    fn test_continuity_at_lambda_zero() {
        let x = 2.0;
        let small_lambda = 1e-12;

        let result_zero = boxcox(x, 0.0).unwrap();
        let result_small = boxcox(x, small_lambda).unwrap();

        // Should be very close
        assert!((result_zero - result_small).abs() < 1e-6);
    }
}
