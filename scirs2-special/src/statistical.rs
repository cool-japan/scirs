//! Statistical convenience functions with comprehensive mathematical foundations
//!
//! This module provides various statistical functions commonly used in
//! machine learning, statistics, and numerical computing, with detailed
//! mathematical theory, derivations, and numerical stability analysis.
//!
//! ## Mathematical Theory and Foundations
//!
//! ### Historical Context
//!
//! These functions form the mathematical backbone of modern machine learning
//! and statistical inference. The logistic function was first introduced by
//! Pierre François Verhulst (1838) for modeling population growth, while the
//! softmax function emerged from the development of statistical mechanics
//! and was later adopted for neural networks by John Hopfield (1982).
//!
//! ### The Logistic Function
//!
//! **Definition**: The logistic function (sigmoid) is defined as:
//! ```text
//! σ(x) = 1 / (1 + e^(-x)) = e^x / (1 + e^x)
//! ```
//!
//! **Mathematical Properties**:
//!
//! 1. **Range**: σ(x) ∈ (0, 1) for all x ∈ ℝ
//!    - **Proof**: Since e^(-x) > 0 for all x, we have 1 + e^(-x) > 1,
//!      so 0 < σ(x) < 1. The limits are: lim_{x→-∞} σ(x) = 0, lim_{x→∞} σ(x) = 1
//!
//! 2. **Symmetry**: σ(-x) = 1 - σ(x)
//!    - **Proof**: σ(-x) = 1/(1 + e^x) = e^(-x)/(e^(-x) + 1) = 1 - 1/(1 + e^(-x)) = 1 - σ(x)
//!
//! 3. **Monotonicity**: σ'(x) = σ(x)(1 - σ(x)) > 0
//!    - **Proof**: d/dx[1/(1 + e^(-x))] = e^(-x)/(1 + e^(-x))² = σ(x) · σ(-x) = σ(x)(1 - σ(x))
//!    - Since σ(x) ∈ (0,1), the derivative is always positive, so σ is strictly increasing
//!
//! 4. **Inflection Point**: The function has an inflection point at x = 0
//!    - **Proof**: σ''(x) = σ'(x)(1 - 2σ(x)) = 0 when σ(x) = 1/2, i.e., at x = 0
//!
//! ### The Softmax Function
//!
//! **Definition**: For a vector x = (x₁, x₂, ..., xₙ), the softmax function is:
//! ```text
//! softmax(xᵢ) = exp(xᵢ) / Σⱼ exp(xⱼ)
//! ```
//!
//! **Mathematical Properties**:
//!
//! 1. **Probability Distribution**: Σᵢ softmax(xᵢ) = 1 and softmax(xᵢ) > 0
//!    - **Proof**: Direct from definition, since we normalize by the sum
//!
//! 2. **Translation Invariance**: softmax(x + c) = softmax(x) for any constant c
//!    - **Proof**: exp(xᵢ + c) / Σⱼ exp(xⱼ + c) = exp(c)exp(xᵢ) / exp(c)Σⱼ exp(xⱼ) = exp(xᵢ) / Σⱼ exp(xⱼ)
//!    - **Numerical Significance**: This property enables the numerically stable
//!      implementation by subtracting max(x) to prevent overflow
//!
//! 3. **Maximum Preservation**: If xₖ > xᵢ for all i ≠ k, then softmax(xₖ) > softmax(xᵢ)
//!    - **Proof**: The exponential function preserves order, and normalization maintains it
//!
//! ### Numerical Stability Analysis
//!
//! **Logistic Function Stability**:
//! - For x ≥ 0: Use σ(x) = 1/(1 + e^(-x)) to avoid exp(x) overflow
//! - For x < 0: Use σ(x) = e^x/(1 + e^x) to avoid e^(-x) underflow
//! - **Error Analysis**: Relative error is O(ε) where ε is machine epsilon
//!
//! **Softmax Stability**:
//! - Standard implementation suffers from overflow when max(x) >> 0
//! - **Solution**: Compute softmax(x - max(x)) using translation invariance
//! - **Condition Number**: κ ≈ max(xᵢ) - min(xᵢ), well-conditioned when differences are moderate
//!
//! ### The LogSumExp Function
//!
//! **Definition**: LSE(x) = log(Σᵢ exp(xᵢ))
//!
//! **Connection to Softmax**: log(softmax(xᵢ)) = xᵢ - LSE(x)
//!
//! **Numerical Implementation**:
//! ```text
//! LSE(x) = max(x) + log(Σᵢ exp(xᵢ - max(x)))
//! ```
//!
//! **Properties**:
//! 1. **Smooth Maximum**: LSE(x) ≥ max(x) with equality as differences grow
//! 2. **Convexity**: LSE is convex (log-sum-exp is log of sum of convex exponentials)
//! 3. **Translation Invariance**: LSE(x + c) = LSE(x) + c
//!
//! ### The Sinc Function
//!
//! **Definition**: The normalized sinc function is:
//! ```text
//! sinc(x) = sin(πx) / (πx) for x ≠ 0, sinc(0) = 1
//! ```
//!
//! **Properties**:
//! 1. **Limit**: lim_{x→0} sin(πx)/(πx) = 1 (by L'Hôpital's rule)
//! 2. **Zeros**: sinc(n) = 0 for all non-zero integers n
//! 3. **Fourier Connection**: The Fourier transform of a rectangular pulse is a sinc function
//! 4. **Sampling Theory**: Central to the Whittaker-Shannon interpolation formula
//!
//! ## Applications in Machine Learning
//!
//! - **Logistic Regression**: The logistic function serves as the link function
//! - **Neural Networks**: Sigmoid activation function in hidden layers
//! - **Softmax Classification**: Output layer for multi-class classification
//! - **Attention Mechanisms**: Softmax for computing attention weights
//! - **Loss Functions**: Cross-entropy loss uses log-softmax for numerical stability

use crate::error::{SpecialError, SpecialResult};
use scirs2_core::ndarray::{Array1, ArrayView1};
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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
/// use scirs2_core::ndarray::array;
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
#[allow(dead_code)]
pub fn softmax(x: ArrayView1<f64>) -> SpecialResult<Array1<f64>> {
    if x.is_empty() {
        return Err(SpecialError::DomainError(
            "Input array cannot be empty".to_string(),
        ));
    }

    // Find maximum for numerical stability
    let xmax = x.fold(f64::NEG_INFINITY, |acc, &val| acc.max(val));

    // Handle case where all values are -infinity
    if xmax == f64::NEG_INFINITY {
        return Err(SpecialError::DomainError(
            "All input values are negative infinity".to_string(),
        ));
    }

    // Compute exp(x_i - xmax)
    let exp_shifted: Array1<f64> = x.mapv(|val| (val - xmax).exp());

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
/// use scirs2_core::ndarray::array;
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
#[allow(dead_code)]
pub fn log_softmax(x: ArrayView1<f64>) -> SpecialResult<Array1<f64>> {
    if x.is_empty() {
        return Err(SpecialError::DomainError(
            "Input array cannot be empty".to_string(),
        ));
    }

    // Find maximum for numerical stability
    let xmax = x.fold(f64::NEG_INFINITY, |acc, &val| acc.max(val));

    // Handle case where all values are -infinity
    if xmax == f64::NEG_INFINITY {
        return Err(SpecialError::DomainError(
            "All input values are negative infinity".to_string(),
        ));
    }

    // Compute log(sum(exp(x_i - xmax))) + xmax
    let log_sum_exp = x.fold(0.0, |acc, &val| acc + (val - xmax).exp()).ln() + xmax;

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
/// use scirs2_core::ndarray::array;
/// use approx::assert_relative_eq;
///
/// let x = array![1.0, 2.0, 3.0];
/// let result = logsumexp(x.view()).unwrap();
///
/// // Should be equivalent to log(exp(1) + exp(2) + exp(3))
/// let manual = (1.0_f64.exp() + 2.0_f64.exp() + 3.0_f64.exp()).ln();
/// assert_relative_eq!(result, manual, epsilon = 1e-10);
/// ```
#[allow(dead_code)]
pub fn logsumexp(x: ArrayView1<f64>) -> SpecialResult<f64> {
    if x.is_empty() {
        return Err(SpecialError::DomainError(
            "Input array cannot be empty".to_string(),
        ));
    }

    // Find maximum for numerical stability
    let xmax = x.fold(f64::NEG_INFINITY, |acc, &val| acc.max(val));

    // Handle case where all values are -infinity
    if xmax == f64::NEG_INFINITY {
        return Ok(f64::NEG_INFINITY);
    }

    // Handle case where maximum is +infinity
    if xmax == f64::INFINITY {
        return Ok(f64::INFINITY);
    }

    // Compute sum(exp(x_i - xmax))
    let sum_exp = x.fold(0.0, |acc, &val| acc + (val - xmax).exp());

    // Return log(sum_exp) + xmax
    Ok(sum_exp.ln() + xmax)
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
/// use scirs2_core::ndarray::array;
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
#[allow(dead_code)]
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
/// use scirs2_core::ndarray::array;
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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
/// use scirs2_core::ndarray::array;
/// use approx::assert_relative_eq;
///
/// let x = array![0.0, 0.5, 1.0, 2.0];
/// let result = sinc_array(x.view());
///
/// assert_eq!(result[0], 1.0);
/// assert_relative_eq!(result[2], 0.0, epsilon = 1e-10);
/// ```
#[allow(dead_code)]
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
#[allow(dead_code)]
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
    use scirs2_core::ndarray::array;

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
