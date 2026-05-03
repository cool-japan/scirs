//! Activation functions for neural networks
//!
//! This module provides standalone activation functions that operate on ndarray Arrays.
//! These functions are designed to work seamlessly with both direct ndarray operations
//! and scirs2-autograd Tensors.

use crate::error::Result;
use scirs2_core::ndarray::{Array, Axis, IxDyn, Zip};
use scirs2_core::numeric::{Float, NumAssign};
use std::fmt::Debug;

/// ReLU (Rectified Linear Unit) activation function
///
/// Applies the element-wise function: `f(x) = max(0, x)`
///
/// # Arguments
///
/// * `input` - Input array
///
/// # Returns
///
/// Array with ReLU applied element-wise
///
/// # Examples
///
/// ```
/// use scirs2_neural::ops::activations::relu;
/// use scirs2_core::ndarray::Array;
///
/// let input = Array::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]).into_dyn();
/// let output = relu(&input).expect("ReLU failed");
/// ```
pub fn relu<F: Float + Debug + NumAssign>(input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
    let mut output = input.clone();
    let zero = F::zero();
    Zip::from(&mut output).for_each(|x| {
        if *x < zero {
            *x = zero;
        }
    });
    Ok(output)
}

/// Sigmoid activation function
///
/// Applies the element-wise function: `f(x) = 1 / (1 + exp(-x))`
///
/// # Arguments
///
/// * `input` - Input array
///
/// # Returns
///
/// Array with sigmoid applied element-wise
///
/// # Examples
///
/// ```
/// use scirs2_neural::ops::activations::sigmoid;
/// use scirs2_core::ndarray::Array;
///
/// let input = Array::from_vec(vec![-1.0, 0.0, 1.0]).into_dyn();
/// let output = sigmoid(&input).expect("Sigmoid failed");
/// ```
pub fn sigmoid<F: Float + Debug + NumAssign>(input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
    let mut output = input.clone();
    let one = F::one();
    Zip::from(&mut output).for_each(|x| {
        *x = one / (one + (-*x).exp());
    });
    Ok(output)
}

/// Tanh (Hyperbolic Tangent) activation function
///
/// Applies the element-wise function: `f(x) = tanh(x)`
///
/// # Arguments
///
/// * `input` - Input array
///
/// # Returns
///
/// Array with tanh applied element-wise
///
/// # Examples
///
/// ```
/// use scirs2_neural::ops::activations::tanh;
/// use scirs2_core::ndarray::Array;
///
/// let input = Array::from_vec(vec![-1.0, 0.0, 1.0]).into_dyn();
/// let output = tanh(&input).expect("Tanh failed");
/// ```
pub fn tanh<F: Float + Debug + NumAssign>(input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
    let mut output = input.clone();
    Zip::from(&mut output).for_each(|x| {
        *x = x.tanh();
    });
    Ok(output)
}

/// GELU (Gaussian Error Linear Unit) activation function
///
/// Applies the approximation: `f(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`
///
/// # Arguments
///
/// * `input` - Input array
///
/// # Returns
///
/// Array with GELU applied element-wise
///
/// # Examples
///
/// ```
/// use scirs2_neural::ops::activations::gelu;
/// use scirs2_core::ndarray::Array;
///
/// let input = Array::from_vec(vec![-1.0, 0.0, 1.0]).into_dyn();
/// let output = gelu(&input).expect("GELU failed");
/// ```
pub fn gelu<F: Float + Debug + NumAssign>(input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
    let mut output = input.clone();
    let half = F::from(0.5).ok_or_else(|| {
        crate::error::NeuralError::ComputationError("Failed to convert constant".to_string())
    })?;
    let sqrt_2_over_pi = F::from(0.7978845608028654).ok_or_else(|| {
        crate::error::NeuralError::ComputationError("Failed to convert constant".to_string())
    })?;
    let coeff = F::from(0.044715).ok_or_else(|| {
        crate::error::NeuralError::ComputationError("Failed to convert constant".to_string())
    })?;
    let one = F::one();

    Zip::from(&mut output).for_each(|x| {
        let x3 = *x * *x * *x;
        let inner = sqrt_2_over_pi * (*x + coeff * x3);
        *x = half * *x * (one + inner.tanh());
    });
    Ok(output)
}

/// Leaky ReLU activation function
///
/// Applies the element-wise function: `f(x) = x if x > 0, else negative_slope * x`
///
/// # Arguments
///
/// * `input` - Input array
/// * `negative_slope` - Slope for negative values (typically 0.01)
///
/// # Returns
///
/// Array with Leaky ReLU applied element-wise
///
/// # Examples
///
/// ```
/// use scirs2_neural::ops::activations::leaky_relu;
/// use scirs2_core::ndarray::Array;
///
/// let input = Array::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]).into_dyn();
/// let output = leaky_relu(&input, 0.01).expect("Leaky ReLU failed");
/// ```
pub fn leaky_relu<F: Float + Debug + NumAssign>(
    input: &Array<F, IxDyn>,
    negative_slope: F,
) -> Result<Array<F, IxDyn>> {
    let mut output = input.clone();
    let zero = F::zero();
    Zip::from(&mut output).for_each(|x| {
        if *x < zero {
            *x = negative_slope * *x;
        }
    });
    Ok(output)
}

/// Swish activation function (also known as SiLU)
///
/// Applies the element-wise function: `f(x) = x * sigmoid(x)`
///
/// # Arguments
///
/// * `input` - Input array
///
/// # Returns
///
/// Array with Swish applied element-wise
///
/// # Examples
///
/// ```
/// use scirs2_neural::ops::activations::swish;
/// use scirs2_core::ndarray::Array;
///
/// let input = Array::from_vec(vec![-1.0, 0.0, 1.0]).into_dyn();
/// let output = swish(&input).expect("Swish failed");
/// ```
pub fn swish<F: Float + Debug + NumAssign>(input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
    let sigmoid_output = sigmoid(input)?;
    let mut output = input.clone();
    Zip::from(&mut output)
        .and(&sigmoid_output)
        .for_each(|x, &sig| {
            *x *= sig;
        });
    Ok(output)
}

/// Mish activation function
///
/// Applies the element-wise function: `f(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))`
///
/// # Arguments
///
/// * `input` - Input array
///
/// # Returns
///
/// Array with Mish applied element-wise
///
/// # Examples
///
/// ```
/// use scirs2_neural::ops::activations::mish;
/// use scirs2_core::ndarray::Array;
///
/// let input = Array::from_vec(vec![-1.0, 0.0, 1.0]).into_dyn();
/// let output = mish(&input).expect("Mish failed");
/// ```
pub fn mish<F: Float + Debug + NumAssign>(input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
    let mut output = input.clone();
    let one = F::one();
    Zip::from(&mut output).for_each(|x| {
        // softplus(x) = ln(1 + exp(x))
        let softplus = (one + x.exp()).ln();
        *x *= softplus.tanh();
    });
    Ok(output)
}

/// Softmax activation function
///
/// Applies softmax along the specified axis: `f(x_i) = exp(x_i) / sum(exp(x_j))`
///
/// # Arguments
///
/// * `input` - Input array
/// * `axis` - Axis along which to apply softmax (-1 for last axis)
///
/// # Returns
///
/// Array with softmax applied along the specified axis
///
/// # Examples
///
/// ```
/// use scirs2_neural::ops::activations::softmax;
/// use scirs2_core::ndarray::Array;
///
/// let input = Array::from_vec(vec![1.0, 2.0, 3.0]).into_dyn();
/// let output = softmax(&input, -1).expect("Softmax failed");
/// ```
pub fn softmax<F: Float + Debug + NumAssign>(
    input: &Array<F, IxDyn>,
    axis: isize,
) -> Result<Array<F, IxDyn>> {
    // For simple 1D case or applying to last axis
    let mut output = input.clone();

    // Determine the actual axis
    let actual_axis = if axis < 0 {
        (input.ndim() as isize + axis) as usize
    } else {
        axis as usize
    };

    if actual_axis >= input.ndim() {
        return Err(crate::error::NeuralError::InvalidArgument(format!(
            "Axis {} out of bounds for array with {} dimensions",
            axis,
            input.ndim()
        )));
    }

    if input.ndim() == 1 {
        // 1D: single-pass max-then-normalise (covers axis 0 and axis -1)
        let max_val = input.fold(F::neg_infinity(), |acc, &x| if x > acc { x } else { acc });
        Zip::from(&mut output).for_each(|x| {
            *x = (*x - max_val).exp();
        });
        let sum = output.sum();
        Zip::from(&mut output).for_each(|x| {
            *x /= sum;
        });
    } else {
        // General multi-dimensional case.
        // Strategy: compute max and sum along `actual_axis`, then broadcast back.
        //
        // `map_axis(Axis(a), f)` collapses axis `a`, so the result shape has
        // `actual_axis` removed.  We re-insert it as a size-1 axis before
        // broadcasting so that Zip can match element-wise.

        // Step 1: compute per-lane max for numerical stability
        let max_vals = input.map_axis(Axis(actual_axis), |view| {
            view.fold(F::neg_infinity(), |a, &b| if b > a { b } else { a })
        });
        // Reinsert the reduced axis as size-1 for broadcasting
        let max_broadcast = max_vals.view().insert_axis(Axis(actual_axis));

        // Step 2: exp(x - max)
        Zip::from(&mut output)
            .and_broadcast(&max_broadcast)
            .for_each(|v, &m| {
                *v = (*v - m).exp();
            });

        // Step 3: compute normalising sum along axis
        let sum_vals = output.map_axis(Axis(actual_axis), |view| {
            view.fold(F::zero(), |a, &b| a + b)
        });
        let sum_broadcast = sum_vals.view().insert_axis(Axis(actual_axis));

        // Step 4: divide by sum
        Zip::from(&mut output)
            .and_broadcast(&sum_broadcast)
            .for_each(|v, &s| {
                *v /= s;
            });
    }

    Ok(output)
}

/// ELU (Exponential Linear Unit) activation function
///
/// Applies the element-wise function: `f(x) = x if x > 0, else alpha * (exp(x) - 1)`
///
/// # Arguments
///
/// * `input` - Input array
/// * `alpha` - Scaling factor for negative values (typically 1.0)
///
/// # Returns
///
/// Array with ELU applied element-wise
///
/// # Examples
///
/// ```
/// use scirs2_neural::ops::activations::elu;
/// use scirs2_core::ndarray::Array;
///
/// let input = Array::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]).into_dyn();
/// let output = elu(&input, 1.0).expect("ELU failed");
/// ```
pub fn elu<F: Float + Debug + NumAssign>(
    input: &Array<F, IxDyn>,
    alpha: F,
) -> Result<Array<F, IxDyn>> {
    let mut output = input.clone();
    let zero = F::zero();
    let one = F::one();
    Zip::from(&mut output).for_each(|x| {
        if *x > zero {
            // x stays the same
        } else {
            *x = alpha * (x.exp() - one);
        }
    });
    Ok(output)
}

/// SELU (Scaled Exponential Linear Unit) activation function
///
/// Applies the element-wise function with self-normalizing properties:
/// `f(x) = scale * (x if x > 0, else alpha * (exp(x) - 1))`
///
/// Default values: scale = 1.0507, alpha = 1.67326
///
/// # Arguments
///
/// * `input` - Input array
///
/// # Returns
///
/// Array with SELU applied element-wise
///
/// # Examples
///
/// ```
/// use scirs2_neural::ops::activations::selu;
/// use scirs2_core::ndarray::Array;
///
/// let input = Array::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]).into_dyn();
/// let output = selu(&input).expect("SELU failed");
/// ```
pub fn selu<F: Float + Debug + NumAssign>(input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
    let scale = F::from(1.050_700_987_355_480_4_f64).ok_or_else(|| {
        crate::error::NeuralError::ComputationError("Failed to convert constant".to_string())
    })?;
    let alpha = F::from(1.673_263_242_354_377_3_f64).ok_or_else(|| {
        crate::error::NeuralError::ComputationError("Failed to convert constant".to_string())
    })?;

    let mut output = input.clone();
    let zero = F::zero();
    let one = F::one();
    Zip::from(&mut output).for_each(|x| {
        if *x > zero {
            *x = scale * *x;
        } else {
            *x = scale * alpha * (x.exp() - one);
        }
    });
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_relu() {
        let input = Array::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]).into_dyn();
        let output = relu(&input).expect("ReLU failed");
        let expected = Array::from_vec(vec![0.0, 0.0, 0.0, 1.0, 2.0]).into_dyn();
        assert_eq!(output, expected);
    }

    #[test]
    fn test_sigmoid() {
        let input = Array::from_vec(vec![0.0]).into_dyn();
        let output = sigmoid(&input).expect("Sigmoid failed");
        assert!((output[[0]] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_tanh() {
        let input = Array::from_vec(vec![0.0]).into_dyn();
        let output = tanh(&input).expect("Tanh failed");
        assert_eq!(output[[0]], 0.0);
    }

    #[test]
    fn test_gelu() {
        let input = Array::from_vec(vec![0.0]).into_dyn();
        let output = gelu(&input).expect("GELU failed");
        assert_eq!(output[[0]], 0.0);
    }

    #[test]
    fn test_leaky_relu() {
        let input = Array::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]).into_dyn();
        let output = leaky_relu(&input, 0.01).expect("Leaky ReLU failed");
        assert!((output[[0]] - (-0.02)).abs() < 1e-6);
        assert!((output[[1]] - (-0.01)).abs() < 1e-6);
        assert_eq!(output[[2]], 0.0);
        assert_eq!(output[[3]], 1.0);
        assert_eq!(output[[4]], 2.0);
    }

    #[test]
    fn test_swish() {
        let input = Array::from_vec(vec![0.0]).into_dyn();
        let output = swish(&input).expect("Swish failed");
        assert_eq!(output[[0]], 0.0);
    }

    #[test]
    fn test_mish() {
        let input = Array::from_vec(vec![0.0]).into_dyn();
        let output = mish(&input).expect("Mish failed");
        assert_eq!(output[[0]], 0.0);
    }

    #[test]
    fn test_softmax_1d() {
        let input = Array::from_vec(vec![1.0, 2.0, 3.0]).into_dyn();
        let output = softmax(&input, -1).expect("Softmax failed");
        // Sum should be 1.0
        let sum: f64 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Values should be increasing
        assert!(output[[0]] < output[[1]]);
        assert!(output[[1]] < output[[2]]);
    }

    #[test]
    fn test_elu() {
        let input = Array::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]).into_dyn();
        let output = elu(&input, 1.0).expect("ELU failed");
        // Positive values should remain unchanged
        assert_eq!(output[[3]], 1.0);
        assert_eq!(output[[4]], 2.0);
        // Negative values should be transformed
        assert!(output[[0]] < 0.0 && output[[0]] > -2.0);
    }

    #[test]
    fn test_selu() {
        let input = Array::from_vec(vec![-1.0, 0.0, 1.0]).into_dyn();
        let output = selu(&input).expect("SELU failed");
        // At 0, should be 0
        assert_eq!(output[[1]], 0.0);
        // Positive values should be scaled
        assert!(output[[2]] > 1.0);
    }

    #[test]
    fn test_softmax_2d_axis0() {
        // Shape: (3, 2) — apply softmax along axis 0
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let arr =
            scirs2_core::ndarray::Array::from_shape_vec(scirs2_core::ndarray::IxDyn(&[3, 2]), data)
                .expect("shape failed");

        let out = softmax(&arr, 0).expect("softmax axis=0 failed");
        assert_eq!(out.shape(), arr.shape());

        // Each column sum should be ~1.0
        for col in 0..2 {
            let col_sum: f64 = (0..3).map(|row| out[[row, col]]).sum();
            assert!(
                (col_sum - 1.0).abs() < 1e-6,
                "column {} sum = {} (expected 1.0)",
                col,
                col_sum
            );
        }
    }

    #[test]
    fn test_softmax_2d_axis1() {
        // Shape: (2, 3) — apply softmax along axis 1 (same as last axis)
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let arr =
            scirs2_core::ndarray::Array::from_shape_vec(scirs2_core::ndarray::IxDyn(&[2, 3]), data)
                .expect("shape failed");

        let out = softmax(&arr, 1).expect("softmax axis=1 failed");
        assert_eq!(out.shape(), arr.shape());

        // Each row sum should be ~1.0
        for row in 0..2 {
            let row_sum: f64 = (0..3).map(|col| out[[row, col]]).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-6,
                "row {} sum = {} (expected 1.0)",
                row,
                row_sum
            );
        }
    }

    #[test]
    fn test_softmax_3d_middle_axis() {
        // Shape: (2, 3, 4) — apply softmax along axis 1
        use scirs2_core::ndarray::Array;
        let arr = Array::from_elem(scirs2_core::ndarray::IxDyn(&[2, 3, 4]), 1.0_f64);
        let out = softmax(&arr, 1).expect("softmax 3d axis=1 failed");
        assert_eq!(out.shape(), arr.shape());
        // Uniform input → each element should be 1/3
        for &v in out.iter() {
            assert!((v - 1.0 / 3.0).abs() < 1e-6);
        }
    }
}
