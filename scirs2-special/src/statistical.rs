//! Statistical convenience functions
//!
//! This module provides various statistical functions commonly used in
//! machine learning, statistics, and numerical computing including:
//! - Logistic function and its derivatives
//! - Softmax and log-softmax functions
//! - Numerically stable logarithmic operations
//! - Normalized sinc function
//! - LogSumExp for numerical stability

use crate::error::{SpecialError, SpecialResult};
use ndarray::{Array1, ArrayView1};
use std::f64::consts::PI;

/// Computes the logistic (sigmoid) function.
///
/// The logistic function is defined as σ(x) = 1 / (1 + e^(-x)).
/// It maps any real number to a value between 0 and 1, making it
/// useful for binary classification and as an activation function.
///
/// # Arguments
///
/// * `x` - Input value
///
/// # Returns
///
/// * `f64` - The logistic function value σ(x)
///
/// # Examples
///
/// ```
/// use scirs2_special::logistic;
/// use approx::assert_relative_eq;
///
/// assert_relative_eq!(logistic(0.0), 0.5, epsilon = 1e-10);
/// assert_relative_eq!(logistic(1.0), 1.0 / (1.0 + (-1.0_f64).exp()), epsilon = 1e-10);
/// assert_relative_eq!(logistic(-1.0), (-1.0_f64).exp() / (1.0 + (-1.0_f64).exp()), epsilon = 1e-10);
/// ```
pub fn logistic(x: f64) -> f64 {
    // Use numerically stable computation to avoid overflow
    if x >= 0.0 {
        let exp_neg_x = (-x).exp();
        1.0 / (1.0 + exp_neg_x)
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

/// Computes the derivative of the logistic function.
///
/// The derivative of the logistic function σ(x) is σ'(x) = σ(x) × (1 - σ(x)).
/// This is commonly used in backpropagation algorithms.
///
/// # Arguments
///
/// * `x` - Input value
///
/// # Returns
///
/// * `f64` - The derivative of the logistic function at x
///
/// # Examples
///
/// ```
/// use scirs2_special::logistic_derivative;
/// use approx::assert_relative_eq;
///
/// // At x=0, σ'(0) = 0.5 * (1 - 0.5) = 0.25
/// assert_relative_eq!(logistic_derivative(0.0), 0.25, epsilon = 1e-10);
/// ```
pub fn logistic_derivative(x: f64) -> f64 {
    let sigma = logistic(x);
    sigma * (1.0 - sigma)
}

/// Computes the softmax function for a vector of inputs.
///
/// The softmax function is defined as softmax(x_i) = exp(x_i) / Σ_j exp(x_j).
/// It converts a vector of real numbers into a probability distribution.
/// This implementation uses the numerically stable version that subtracts
/// the maximum value to prevent overflow.
///
/// # Arguments
///
/// * `x` - Input array
///
/// # Returns
///
/// * `SpecialResult<Array1<f64>>` - The softmax probabilities
///
/// # Examples
///
/// ```
/// use scirs2_special::softmax;
/// use ndarray::array;
/// use approx::assert_relative_eq;
///
/// let x = array![1.0, 2.0, 3.0];
/// let result = softmax(x.view()).unwrap();
///
/// // Check that probabilities sum to 1
/// assert_relative_eq!(result.sum(), 1.0, epsilon = 1e-10);
///
/// // Check that all values are positive
/// assert!(result.iter().all(|&val| val > 0.0));
/// ```
pub fn softmax(x: ArrayView1<f64>) -> SpecialResult<Array1<f64>> {
    if x.is_empty() {
        return Err(SpecialError::DomainError(
            "Input array cannot be empty".to_string(),
        ));
    }

    // Find maximum for numerical stability
    let x_max = x.fold(f64::NEG_INFINITY, |acc, &val| acc.max(val));

    // Handle case where all values are -infinity
    if x_max == f64::NEG_INFINITY {
        return Err(SpecialError::DomainError(
            "All input values are negative infinity".to_string(),
        ));
    }

    // Compute exp(x_i - x_max)
    let exp_shifted: Array1<f64> = x.mapv(|val| (val - x_max).exp());

    // Compute sum of exponentials
    let sum_exp = exp_shifted.sum();

    // Handle numerical issues
    if sum_exp == 0.0 || !sum_exp.is_finite() {
        return Err(SpecialError::DomainError(
            "Numerical overflow in softmax computation".to_string(),
        ));
    }

    // Normalize to get probabilities
    Ok(exp_shifted / sum_exp)
}

/// Computes the log-softmax function for a vector of inputs.
///
/// The log-softmax function is defined as log_softmax(x_i) = x_i - log(Σ_j exp(x_j)).
/// This is more numerically stable than computing log(softmax(x)) and is commonly
/// used in machine learning for classification problems.
///
/// # Arguments
///
/// * `x` - Input array
///
/// # Returns
///
/// * `SpecialResult<Array1<f64>>` - The log-softmax values
///
/// # Examples
///
/// ```
/// use scirs2_special::log_softmax;
/// use ndarray::array;
/// use approx::assert_relative_eq;
///
/// let x = array![1.0, 2.0, 3.0];
/// let result = log_softmax(x.view()).unwrap();
///
/// // log_softmax should be equivalent to log(softmax(x))
/// let softmax_result = scirs2_special::softmax(x.view()).unwrap();
/// let log_softmax_manual = softmax_result.mapv(|val| val.ln());
///
/// for (a, b) in result.iter().zip(log_softmax_manual.iter()) {
///     assert_relative_eq!(a, b, epsilon = 1e-10);
/// }
/// ```
pub fn log_softmax(x: ArrayView1<f64>) -> SpecialResult<Array1<f64>> {
    if x.is_empty() {
        return Err(SpecialError::DomainError(
            "Input array cannot be empty".to_string(),
        ));
    }

    // Find maximum for numerical stability
    let x_max = x.fold(f64::NEG_INFINITY, |acc, &val| acc.max(val));

    // Handle case where all values are -infinity
    if x_max == f64::NEG_INFINITY {
        return Err(SpecialError::DomainError(
            "All input values are negative infinity".to_string(),
        ));
    }

    // Compute log(sum(exp(x_i - x_max))) + x_max
    let log_sum_exp = x.fold(0.0, |acc, &val| acc + (val - x_max).exp()).ln() + x_max;

    // Handle numerical issues
    if !log_sum_exp.is_finite() {
        return Err(SpecialError::DomainError(
            "Numerical overflow in log-softmax computation".to_string(),
        ));
    }

    // Compute log_softmax(x_i) = x_i - log_sum_exp
    Ok(x.mapv(|val| val - log_sum_exp))
}

/// Computes the LogSumExp function for numerical stability.
///
/// The LogSumExp function is defined as LSE(x) = log(Σ_i exp(x_i)).
/// This is computed in a numerically stable way by factoring out the maximum value.
/// This function is useful for avoiding overflow in probabilistic computations.
///
/// # Arguments
///
/// * `x` - Input array
///
/// # Returns
///
/// * `SpecialResult<f64>` - The LogSumExp value
///
/// # Examples
///
/// ```
/// use scirs2_special::logsumexp;
/// use ndarray::array;
/// use approx::assert_relative_eq;
///
/// let x = array![1.0, 2.0, 3.0];
/// let result = logsumexp(x.view()).unwrap();
///
/// // Should be equivalent to log(exp(1) + exp(2) + exp(3))
/// let manual = (1.0_f64.exp() + 2.0_f64.exp() + 3.0_f64.exp()).ln();
/// assert_relative_eq!(result, manual, epsilon = 1e-10);
/// ```
pub fn logsumexp(x: ArrayView1<f64>) -> SpecialResult<f64> {
    if x.is_empty() {
        return Err(SpecialError::DomainError(
            "Input array cannot be empty".to_string(),
        ));
    }

    // Find maximum for numerical stability
    let x_max = x.fold(f64::NEG_INFINITY, |acc, &val| acc.max(val));

    // Handle case where all values are -infinity
    if x_max == f64::NEG_INFINITY {
        return Ok(f64::NEG_INFINITY);
    }

    // Handle case where maximum is +infinity
    if x_max == f64::INFINITY {
        return Ok(f64::INFINITY);
    }

    // Compute sum(exp(x_i - x_max))
    let sum_exp = x.fold(0.0, |acc, &val| acc + (val - x_max).exp());

    // Return log(sum_exp) + x_max
    Ok(sum_exp.ln() + x_max)
}

/// Computes log(1 + x) with improved numerical stability for small x.
///
/// This function provides array support for the log1p operation,
/// which is more accurate than log(1 + x) for small values of x.
///
/// # Arguments
///
/// * `x` - Input array
///
/// # Returns
///
/// * `Array1<f64>` - The log(1 + x) values
///
/// # Examples
///
/// ```
/// use scirs2_special::log1p_array;
/// use ndarray::array;
/// use approx::assert_relative_eq;
///
/// let x = array![0.0, 1e-10, 0.1, 1.0];
/// let result = log1p_array(x.view());
///
/// // Check that it matches std::f64::log1p for individual values
/// for (input, output) in x.iter().zip(result.iter()) {
///     assert_relative_eq!(*output, input.ln_1p(), epsilon = 1e-15);
/// }
/// ```
pub fn log1p_array(x: ArrayView1<f64>) -> Array1<f64> {
    x.mapv(|val| val.ln_1p())
}

/// Computes exp(x) - 1 with improved numerical stability for small x.
///
/// This function provides array support for the expm1 operation,
/// which is more accurate than exp(x) - 1 for small values of x.
///
/// # Arguments
///
/// * `x` - Input array
///
/// # Returns
///
/// * `Array1<f64>` - The exp(x) - 1 values
///
/// # Examples
///
/// ```
/// use scirs2_special::expm1_array;
/// use ndarray::array;
/// use approx::assert_relative_eq;
///
/// let x = array![0.0, 1e-10, 0.1, 1.0];
/// let result = expm1_array(x.view());
///
/// // Check that it matches std::f64::exp_m1 for individual values
/// for (input, output) in x.iter().zip(result.iter()) {
///     assert_relative_eq!(*output, input.exp_m1(), epsilon = 1e-15);
/// }
/// ```
pub fn expm1_array(x: ArrayView1<f64>) -> Array1<f64> {
    x.mapv(|val| val.exp_m1())
}

/// Computes the normalized sinc function.
///
/// The normalized sinc function is defined as sinc(x) = sin(πx) / (πx) for x ≠ 0,
/// and sinc(0) = 1. This is the form commonly used in signal processing.
///
/// # Arguments
///
/// * `x` - Input value
///
/// # Returns
///
/// * `f64` - The normalized sinc function value
///
/// # Examples
///
/// ```
/// use scirs2_special::sinc;
/// use approx::assert_relative_eq;
///
/// assert_eq!(sinc(0.0), 1.0);
/// assert_relative_eq!(sinc(1.0), 0.0, epsilon = 1e-10);
/// assert_relative_eq!(sinc(0.5), 2.0 / std::f64::consts::PI, epsilon = 1e-10);
/// ```
pub fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let pi_x = PI * x;
        pi_x.sin() / pi_x
    }
}

/// Computes the normalized sinc function for an array of inputs.
///
/// This is the vectorized version of the sinc function.
///
/// # Arguments
///
/// * `x` - Input array
///
/// # Returns
///
/// * `Array1<f64>` - The sinc function values
///
/// # Examples
///
/// ```
/// use scirs2_special::sinc_array;
/// use ndarray::array;
/// use approx::assert_relative_eq;
///
/// let x = array![0.0, 0.5, 1.0, 2.0];
/// let result = sinc_array(x.view());
///
/// assert_eq!(result[0], 1.0);
/// assert_relative_eq!(result[2], 0.0, epsilon = 1e-10);
/// ```
pub fn sinc_array(x: ArrayView1<f64>) -> Array1<f64> {
    x.mapv(sinc)
}

/// Computes the log of the absolute value of the gamma function.
///
/// This function provides a more numerically stable way to compute log(|Γ(x)|)
/// for large values of x, especially when the gamma function itself would overflow.
///
/// # Arguments
///
/// * `x` - Input value
///
/// # Returns
///
/// * `SpecialResult<f64>` - The log absolute gamma value
///
/// # Examples
///
/// ```
/// use scirs2_special::log_abs_gamma;
/// use approx::assert_relative_eq;
///
/// // For small positive integers, log(Γ(n)) = log((n-1)!)
/// assert_relative_eq!(log_abs_gamma(5.0).unwrap(), (24.0_f64).ln(), epsilon = 1e-10);
/// ```
pub fn log_abs_gamma(x: f64) -> SpecialResult<f64> {
    if x <= 0.0 && x == x.floor() {
        // Gamma function has poles at non-positive integers
        return Err(SpecialError::DomainError(
            "Gamma function undefined at non-positive integers".to_string(),
        ));
    }

    // Use the gammaln function from our gamma module
    use crate::gamma::gammaln;
    Ok(gammaln(x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_logistic() {
        assert_relative_eq!(logistic(0.0), 0.5, epsilon = 1e-10);
        assert_relative_eq!(
            logistic(1.0),
            1.0 / (1.0 + (-1.0_f64).exp()),
            epsilon = 1e-10
        );
        assert_relative_eq!(
            logistic(-1.0),
            (-1.0_f64).exp() / (1.0 + (-1.0_f64).exp()),
            epsilon = 1e-10
        );

        // Test extreme values
        assert!(logistic(100.0) > 0.99);
        assert!(logistic(-100.0) < 0.01);
    }

    #[test]
    fn test_logistic_derivative() {
        assert_relative_eq!(logistic_derivative(0.0), 0.25, epsilon = 1e-10);

        // Test that derivative is always positive
        for &x in &[-5.0, -1.0, 0.0, 1.0, 5.0] {
            assert!(logistic_derivative(x) > 0.0);
        }
    }

    #[test]
    fn test_softmax() {
        let x = array![1.0, 2.0, 3.0];
        let result = softmax(x.view()).unwrap();

        // Check that probabilities sum to 1
        assert_relative_eq!(result.sum(), 1.0, epsilon = 1e-10);

        // Check that all values are positive
        assert!(result.iter().all(|&val| val > 0.0));

        // Check that larger inputs get larger probabilities
        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }

    #[test]
    fn test_log_softmax() {
        let x = array![1.0, 2.0, 3.0];
        let result = log_softmax(x.view()).unwrap();

        // log_softmax should be equivalent to log(softmax(x))
        let softmax_result = softmax(x.view()).unwrap();
        let log_softmax_manual = softmax_result.mapv(|val| val.ln());

        for (a, b) in result.iter().zip(log_softmax_manual.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_logsumexp() {
        let x = array![1.0, 2.0, 3.0];
        let result = logsumexp(x.view()).unwrap();

        // Should be equivalent to log(exp(1) + exp(2) + exp(3))
        let manual = (1.0_f64.exp() + 2.0_f64.exp() + 3.0_f64.exp()).ln();
        assert_relative_eq!(result, manual, epsilon = 1e-10);

        // Test with large values to check numerical stability
        let large_x = array![1000.0, 1001.0, 1002.0];
        let large_result = logsumexp(large_x.view()).unwrap();
        assert!(large_result.is_finite());
    }

    #[test]
    fn test_log1p_array() {
        let x = array![0.0, 1e-10, 0.1, 1.0];
        let result = log1p_array(x.view());

        for (input, output) in x.iter().zip(result.iter()) {
            assert_relative_eq!(*output, input.ln_1p(), epsilon = 1e-15);
        }
    }

    #[test]
    fn test_expm1_array() {
        let x = array![0.0, 1e-10, 0.1, 1.0];
        let result = expm1_array(x.view());

        for (input, output) in x.iter().zip(result.iter()) {
            assert_relative_eq!(*output, input.exp_m1(), epsilon = 1e-15);
        }
    }

    #[test]
    fn test_sinc() {
        assert_eq!(sinc(0.0), 1.0);
        assert_relative_eq!(sinc(1.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(sinc(0.5), 2.0 / PI, epsilon = 1e-10);
        assert_relative_eq!(sinc(-1.0), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sinc_array() {
        let x = array![0.0, 0.5, 1.0, 2.0];
        let result = sinc_array(x.view());

        assert_eq!(result[0], 1.0);
        assert_relative_eq!(result[2], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_log_abs_gamma() {
        // For small positive integers, log(Γ(n)) = log((n-1)!)
        assert_relative_eq!(log_abs_gamma(1.0).unwrap(), 0.0, epsilon = 1e-10); // log(0!) = log(1)
        assert_relative_eq!(log_abs_gamma(2.0).unwrap(), 0.0, epsilon = 1e-10); // log(1!) = log(1)
        assert_relative_eq!(log_abs_gamma(3.0).unwrap(), (2.0_f64).ln(), epsilon = 1e-10); // log(2!)
        assert_relative_eq!(
            log_abs_gamma(5.0).unwrap(),
            (24.0_f64).ln(),
            epsilon = 1e-10
        ); // log(4!)

        // Test error for non-positive integers
        assert!(log_abs_gamma(0.0).is_err());
        assert!(log_abs_gamma(-1.0).is_err());
        assert!(log_abs_gamma(-2.0).is_err());
    }
}
