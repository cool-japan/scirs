//! Matrix multiplication operations for batch processing in neural networks
//!
//! This module contains optimized functions for batch matrix multiplication operations
//! that are commonly used in neural network computations.

use ndarray::{s, Array2, Array3, ArrayView2, ArrayView3};
use num_traits::Float;
use std::fmt::Debug;

use crate::error::{NeuralError, Result};

/// Perform batch matrix multiplication for neural network operations.
///
/// This function multiplies batches of matrices efficiently, which is common
/// in neural network computations like batch processing of fully connected layers.
///
/// # Arguments
///
/// * `a` - First batch of matrices with shape [batch_size, m, k]
/// * `b` - Second batch of matrices with shape [batch_size, k, n]
///
/// # Returns
///
/// * Result matrix with shape [batch_size, m, n]
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array, Ix3};
/// use scirs2_neural::linalg::batch_matmul;
///
/// // Create batch of 2x2x3 matrices (batch_size=2, m=2, k=3)
/// let a = Array::from_shape_vec(
///     (2, 2, 3),
///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
/// ).unwrap();
///
/// // Create batch of 2x3x2 matrices (batch_size=2, k=3, n=2)
/// let b = Array::from_shape_vec(
///     (2, 3, 2),
///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
/// ).unwrap();
///
/// // Result should be shape [2, 2, 2]
/// let c = batch_matmul(&a.view(), &b.view()).unwrap();
/// assert_eq!(c.shape(), &[2, 2, 2]);
/// ```
pub fn batch_matmul<F>(a: &ArrayView3<F>, b: &ArrayView3<F>) -> Result<Array3<F>>
where
    F: Float + Debug,
{
    // Get dimensions
    let batch_size = a.shape()[0];
    if batch_size != b.shape()[0] {
        return Err(NeuralError::ShapeMismatch(format!(
            "Batch size mismatch in batch_matmul: a batch_size={}, b batch_size={}",
            batch_size,
            b.shape()[0]
        )));
    }

    let m = a.shape()[1];
    let k = a.shape()[2];
    let k2 = b.shape()[1];
    let n = b.shape()[2];

    if k != k2 {
        return Err(NeuralError::ShapeMismatch(format!(
            "Inner dimensions mismatch in batch_matmul: a has k={}, b has k={}",
            k, k2
        )));
    }

    // Allocate output array
    let mut result = Array3::<F>::zeros((batch_size, m, n));

    // Perform batch matrix multiplication
    for batch_idx in 0..batch_size {
        let a_slice = a.slice(s![batch_idx, .., ..]);
        let b_slice = b.slice(s![batch_idx, .., ..]);

        for i in 0..m {
            for j in 0..n {
                let mut sum = F::zero();

                for l in 0..k {
                    sum = sum + a_slice[[i, l]] * b_slice[[l, j]];
                }

                result[[batch_idx, i, j]] = sum;
            }
        }
    }

    Ok(result)
}

/// Perform batch vector-matrix multiplication for neural network operations.
///
/// This is commonly used in RNN and attention mechanisms.
///
/// # Arguments
///
/// * `v` - Batch of vectors with shape [batch_size, k]
/// * `m` - Batch of matrices with shape [batch_size, k, n]
///
/// # Returns
///
/// * Result batch of vectors with shape [batch_size, n]
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array, Ix2, Ix3};
/// use scirs2_neural::linalg::batch_vecmat;
///
/// // Create batch of 2x3 vectors (batch_size=2, k=3)
/// let v = Array::from_shape_vec(
///     (2, 3),
///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
/// ).unwrap();
///
/// // Create batch of 2x3x2 matrices (batch_size=2, k=3, n=2)
/// let m = Array::from_shape_vec(
///     (2, 3, 2),
///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
/// ).unwrap();
///
/// // Result should be shape [2, 2]
/// let result = batch_vecmat(&v.view(), &m.view()).unwrap();
/// assert_eq!(result.shape(), &[2, 2]);
/// ```
pub fn batch_vecmat<F>(v: &ArrayView2<F>, m: &ArrayView3<F>) -> Result<Array2<F>>
where
    F: Float + Debug,
{
    // Get dimensions
    let batch_size = v.shape()[0];
    if batch_size != m.shape()[0] {
        return Err(NeuralError::ShapeMismatch(format!(
            "Batch size mismatch in batch_vecmat: v batch_size={}, m batch_size={}",
            batch_size,
            m.shape()[0]
        )));
    }

    let k = v.shape()[1];
    let k2 = m.shape()[1];
    let n = m.shape()[2];

    if k != k2 {
        return Err(NeuralError::ShapeMismatch(format!(
            "Inner dimensions mismatch in batch_vecmat: v has k={}, m has k={}",
            k, k2
        )));
    }

    // Allocate output array
    let mut result = Array2::<F>::zeros((batch_size, n));

    // Perform batch vector-matrix multiplication
    for batch_idx in 0..batch_size {
        let v_slice = v.slice(s![batch_idx, ..]);
        let m_slice = m.slice(s![batch_idx, .., ..]);

        for j in 0..n {
            let mut sum = F::zero();

            for l in 0..k {
                sum = sum + v_slice[l] * m_slice[[l, j]];
            }

            result[[batch_idx, j]] = sum;
        }
    }

    Ok(result)
}

/// Computes the gradient for batch matrix multiplication.
///
/// This function calculates the gradients with respect to inputs `a` and `b`
/// for the batch matrix multiplication operation.
///
/// # Arguments
///
/// * `a` - First batch of matrices with shape [batch_size, m, k]
/// * `b` - Second batch of matrices with shape [batch_size, k, n]
/// * `grad_output` - Gradient of the loss with respect to the output of batch_matmul,
///   with shape [batch_size, m, n]
///
/// # Returns
///
/// * Tuple of (grad_a, grad_b) where:
///   - grad_a has shape [batch_size, m, k]
///   - grad_b has shape [batch_size, k, n]
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array, Ix3};
/// use scirs2_neural::linalg::{batch_matmul, batch_matmul_backward};
///
/// // Create input matrices
/// let a = Array::from_shape_vec(
///     (2, 2, 3),
///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
/// ).unwrap();
///
/// let b = Array::from_shape_vec(
///     (2, 3, 2),
///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
/// ).unwrap();
///
/// // Assume a gradient from the next layer
/// let grad_output = Array::from_shape_vec(
///     (2, 2, 2),
///     vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
/// ).unwrap();
///
/// // Compute gradients
/// let (grad_a, grad_b) = batch_matmul_backward(&a.view(), &b.view(), &grad_output.view()).unwrap();
/// assert_eq!(grad_a.shape(), a.shape());
/// assert_eq!(grad_b.shape(), b.shape());
/// ```
pub fn batch_matmul_backward<F>(
    a: &ArrayView3<F>,
    b: &ArrayView3<F>,
    grad_output: &ArrayView3<F>,
) -> Result<(Array3<F>, Array3<F>)>
where
    F: Float + Debug,
{
    // Get dimensions
    let batch_size = a.shape()[0];
    if batch_size != b.shape()[0] || batch_size != grad_output.shape()[0] {
        return Err(NeuralError::ShapeMismatch(
            format!("Batch size mismatch in batch_matmul_backward: a batch_size={}, b batch_size={}, grad_output batch_size={}", 
                    batch_size, b.shape()[0], grad_output.shape()[0])
        ));
    }

    let m = a.shape()[1];
    let k = a.shape()[2];
    let k2 = b.shape()[1];
    let n = b.shape()[2];

    let m2 = grad_output.shape()[1];
    let n2 = grad_output.shape()[2];

    if k != k2 {
        return Err(NeuralError::ShapeMismatch(format!(
            "Inner dimensions mismatch in batch_matmul_backward: a has k={}, b has k={}",
            k, k2
        )));
    }

    if m != m2 || n != n2 {
        return Err(NeuralError::ShapeMismatch(
            format!("Output dimensions mismatch in batch_matmul_backward: expected [batch_size, {}, {}], got [batch_size, {}, {}]", 
                    m, n, m2, n2)
        ));
    }

    // Allocate gradient arrays
    let mut grad_a = Array3::<F>::zeros((batch_size, m, k));
    let mut grad_b = Array3::<F>::zeros((batch_size, k, n));

    // Compute gradients
    for batch_idx in 0..batch_size {
        let a_slice = a.slice(s![batch_idx, .., ..]);
        let b_slice = b.slice(s![batch_idx, .., ..]);
        let grad_output_slice = grad_output.slice(s![batch_idx, .., ..]);

        // Compute grad_a: grad_output * b^T
        for i in 0..m {
            for l in 0..k {
                let mut sum = F::zero();

                for j in 0..n {
                    sum = sum + grad_output_slice[[i, j]] * b_slice[[l, j]];
                }

                grad_a[[batch_idx, i, l]] = sum;
            }
        }

        // Compute grad_b: a^T * grad_output
        for l in 0..k {
            for j in 0..n {
                let mut sum = F::zero();

                for i in 0..m {
                    sum = sum + a_slice[[i, l]] * grad_output_slice[[i, j]];
                }

                grad_b[[batch_idx, l, j]] = sum;
            }
        }
    }

    Ok((grad_a, grad_b))
}
