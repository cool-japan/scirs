//! Batch matrix operations for AI/ML workloads
//!
//! This module provides matrix operations optimized for processing batches of data,
//! which is especially useful for machine learning applications such as mini-batch
//! gradient descent, convolutional neural networks, and transformer models.

use scirs2_core::ndarray::{
    Array, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis, ScalarOperand,
};
use scirs2_core::numeric::{Float, NumAssign};
use std::iter::Sum;

use crate::error::{LinalgError, LinalgResult};

// Re-export batch attention operations
pub mod attention;
pub use attention::{
    batch_flash_attention, batch_multi_head_attention, batch_multi_query_attention,
};

/// Perform matrix multiplication on a batch of matrices
///
/// Computes a batch multiplication where the input represents multiple matrices
/// that need to be multiplied with the same right-hand side matrix.
///
/// # Arguments
///
/// * `batch_a` - 3D array of shape (batchsize, m, k) representing the batch of matrices
/// * `b` - 2D array of shape (k, n) representing the right-hand side matrix
///
/// # Returns
///
/// * 3D array of shape (batchsize, m, n) containing the result of each batch multiplication
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, Array3};
/// use scirs2_linalg::batch::batch_matmul;
///
/// // Create a batch of 2 matrices, each 2x2
/// let batch_a = Array3::from_shape_vec((2, 2, 2), vec![
///     1.0, 2.0,  // First matrix: [[1.0, 2.0],
///     3.0, 4.0,  //               [3.0, 4.0]]
///     5.0, 6.0,  // Second matrix: [[5.0, 6.0],
///     7.0, 8.0   //                [7.0, 8.0]]
/// ]).unwrap();
///
/// // Create a 2x1 matrix to multiply with each batch matrix
/// let b = array![[10.0], [20.0]];
///
/// // Perform batch multiplication
/// let result = batch_matmul(&batch_a.view(), &b.view()).unwrap();
///
/// // Expected results:
/// // First result: [[1.0, 2.0], [3.0, 4.0]] × [[10.0], [20.0]] = [[50.0], [110.0]]
/// // Second result: [[5.0, 6.0], [7.0, 8.0]] × [[10.0], [20.0]] = [[170.0], [230.0]]
/// assert_eq!(result.shape(), &[2, 2, 1]);
/// assert_eq!(result[[0, 0, 0]], 50.0);
/// assert_eq!(result[[0, 1, 0]], 110.0);
/// assert_eq!(result[[1, 0, 0]], 170.0);
/// assert_eq!(result[[1, 1, 0]], 230.0);
/// ```
#[allow(dead_code)]
pub fn batch_matmul<F>(batch_a: &ArrayView3<F>, b: &ArrayView2<F>) -> LinalgResult<Array3<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    // Check dimensions compatibility
    let (batchsize, m, k1) = batch_a.dim();
    let (k2, n) = b.dim();

    if k1 != k2 {
        return Err(LinalgError::ShapeError(format!(
            "Inner dimensions mismatch for batch_matmul: {k1} vs {k2}"
        )));
    }

    // Initialize result array
    let mut result = Array::zeros((batchsize, m, n));

    // Perform batch matrix multiplication
    for batch_idx in 0..batchsize {
        for i in 0..m {
            for j in 0..n {
                // Compute the dot product between row i of matrix from batch_a and column j of b
                let mut sum = F::zero();
                for k in 0..k1 {
                    sum += batch_a[[batch_idx, i, k]] * b[[k, j]];
                }
                result[[batch_idx, i, j]] = sum;
            }
        }
    }

    Ok(result)
}

/// Batch matrix-vector multiplication
///
/// Performs matrix-vector multiplication for a batch of matrices with a single vector.
///
/// # Arguments
///
/// * `batch_a` - 3D array of shape (batchsize, m, n) representing the batch of matrices
/// * `x` - Vector of length n
///
/// # Returns
///
/// * 2D array of shape (batchsize, m) containing the result of each matrix-vector multiplication
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, Array3};
/// use scirs2_linalg::batch::batch_matvec;
///
/// // Create a batch of 2 matrices, each 2x2
/// let batch_a = Array3::from_shape_vec((2, 2, 2), vec![
///     1.0, 2.0,  // First matrix: [[1.0, 2.0],
///     3.0, 4.0,  //               [3.0, 4.0]]
///     5.0, 6.0,  // Second matrix: [[5.0, 6.0],
///     7.0, 8.0   //                [7.0, 8.0]]
/// ]).unwrap();
///
/// // Create a vector to multiply with each batch matrix
/// let x = array![10.0, 20.0];
///
/// // Perform batch matrix-vector multiplication
/// let result = batch_matvec(&batch_a.view(), &x.view()).unwrap();
///
/// // Expected results:
/// // First result: [[1.0, 2.0], [3.0, 4.0]] × [10.0, 20.0] = [50.0, 110.0]
/// // Second result: [[5.0, 6.0], [7.0, 8.0]] × [10.0, 20.0] = [170.0, 230.0]
/// assert_eq!(result.shape(), &[2, 2]);
/// assert_eq!(result[[0, 0]], 50.0);
/// assert_eq!(result[[0, 1]], 110.0);
/// assert_eq!(result[[1, 0]], 170.0);
/// assert_eq!(result[[1, 1]], 230.0);
/// ```
#[allow(dead_code)]
pub fn batch_matvec<F>(batch_a: &ArrayView3<F>, x: &ArrayView1<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    // Check dimensions compatibility
    let (batchsize, m, n) = batch_a.dim();
    let x_len = x.len();

    if n != x_len {
        return Err(LinalgError::ShapeError(format!(
            "Dimension mismatch for batch_matvec: matrix width {n} does not match vector length {x_len}"
        )));
    }

    // Initialize result array
    let mut result = Array::zeros((batchsize, m));

    // Perform batch matrix-vector multiplication
    for batch_idx in 0..batchsize {
        for i in 0..m {
            // Compute the dot product between row i of matrix from batch_a and vector x
            let mut sum = F::zero();
            for j in 0..n {
                sum += batch_a[[batch_idx, i, j]] * x[j];
            }
            result[[batch_idx, i]] = sum;
        }
    }

    Ok(result)
}

/// Add a vector to each matrix in a batch
///
/// Adds a vector to each row or column of each matrix in the batch,
/// supporting broadcasting.
///
/// # Arguments
///
/// * `batch_a` - 3D array of shape (batchsize, m, n) representing the batch of matrices
/// * `v` - Vector to add to each matrix in the batch
/// * `axis` - Axis along which to add the vector (0 for column-wise, 1 for row-wise)
///
/// # Returns
///
/// * 3D array of shape (batchsize, m, n) containing the result
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, Array3, Axis};
/// use scirs2_linalg::batch::batch_add;
///
/// // Create a batch of 2 matrices, each 2x2
/// let batch_a = Array3::from_shape_vec((2, 2, 2), vec![
///     1.0, 2.0,  // First matrix: [[1.0, 2.0],
///     3.0, 4.0,  //               [3.0, 4.0]]
///     5.0, 6.0,  // Second matrix: [[5.0, 6.0],
///     7.0, 8.0   //                [7.0, 8.0]]
/// ]).unwrap();
///
/// // Create a vector to add to each row of each matrix
/// let v = array![10.0, 20.0];
///
/// // Add to each row (axis=1)
/// let result = batch_add(&batch_a.view(), &v.view(), 1).unwrap();
///
/// // Expected results:
/// // First matrix: [[1.0+10.0, 2.0+20.0], [3.0+10.0, 4.0+20.0]] = [[11.0, 22.0], [13.0, 24.0]]
/// // Second matrix: [[5.0+10.0, 6.0+20.0], [7.0+10.0, 8.0+20.0]] = [[15.0, 26.0], [17.0, 28.0]]
/// assert_eq!(result.shape(), &[2, 2, 2]);
/// assert_eq!(result[[0, 0, 0]], 11.0);
/// assert_eq!(result[[0, 0, 1]], 22.0);
/// assert_eq!(result[[0, 1, 0]], 13.0);
/// assert_eq!(result[[0, 1, 1]], 24.0);
/// assert_eq!(result[[1, 0, 0]], 15.0);
/// assert_eq!(result[[1, 0, 1]], 26.0);
/// assert_eq!(result[[1, 1, 0]], 17.0);
/// assert_eq!(result[[1, 1, 1]], 28.0);
/// ```
#[allow(dead_code)]
pub fn batch_add<F>(
    batch_a: &ArrayView3<F>,
    v: &ArrayView1<F>,
    axis: usize,
) -> LinalgResult<Array3<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    // Check dimensions compatibility
    let (batchsize, m, n) = batch_a.dim();

    // Check if vector dimension is compatible with the specified axis
    match axis {
        0 => {
            // Column-wise addition: vector length should match m
            if v.len() != m {
                return Err(LinalgError::ShapeError(format!(
                    "Column-wise batch_add requires vector length {} to match number of rows {}, got {}",
                    m, m, v.len()
                )));
            }
        }
        1 => {
            // Row-wise addition: vector length should match n
            if v.len() != n {
                return Err(LinalgError::ShapeError(format!(
                    "Row-wise batch_add requires vector length {} to match number of columns {}, got {}",
                    n, n, v.len()
                )));
            }
        }
        _ => {
            return Err(LinalgError::InvalidInputError(format!(
                "Invalid axis {axis}: must be 0 (column-wise) or 1 (row-wise)"
            )));
        }
    }

    // Initialize result array with _a copy of the input batch
    let mut result = batch_a.to_owned();

    // Perform batch addition
    match axis {
        0 => {
            // Column-wise addition (add to each column)
            for batch_idx in 0..batchsize {
                for i in 0..m {
                    for j in 0..n {
                        result[[batch_idx, i, j]] += v[i];
                    }
                }
            }
        }
        1 => {
            // Row-wise addition (add to each row)
            for batch_idx in 0..batchsize {
                for i in 0..m {
                    for j in 0..n {
                        result[[batch_idx, i, j]] += v[j];
                    }
                }
            }
        }
        _ => unreachable!(), // Already validated above
    }

    Ok(result)
}

/// Compute the sum of a batch of matrices
///
/// Sums the matrices in a batch along the batch dimension.
///
/// # Arguments
///
/// * `batch_a` - 3D array of shape (batchsize, m, n) representing the batch of matrices
///
/// # Returns
///
/// * 2D array of shape (m, n) containing the sum of all matrices in the batch
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, Array3};
/// use scirs2_linalg::batch::batch_sum;
///
/// // Create a batch of 2 matrices, each 2x2
/// let batch_a = Array3::from_shape_vec((2, 2, 2), vec![
///     1.0, 2.0,  // First matrix: [[1.0, 2.0],
///     3.0, 4.0,  //               [3.0, 4.0]]
///     5.0, 6.0,  // Second matrix: [[5.0, 6.0],
///     7.0, 8.0   //                [7.0, 8.0]]
/// ]).unwrap();
///
/// // Compute the sum of all matrices in the batch
/// let result = batch_sum(&batch_a.view());
///
/// // Expected result: [[1.0, 2.0], [3.0, 4.0]] + [[5.0, 6.0], [7.0, 8.0]] = [[6.0, 8.0], [10.0, 12.0]]
/// assert_eq!(result.shape(), &[2, 2]);
/// assert_eq!(result[[0, 0]], 6.0);
/// assert_eq!(result[[0, 1]], 8.0);
/// assert_eq!(result[[1, 0]], 10.0);
/// assert_eq!(result[[1, 1]], 12.0);
/// ```
#[allow(dead_code)]
pub fn batch_sum<F>(batch_a: &ArrayView3<F>) -> Array2<F>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    batch_a.sum_axis(Axis(0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::{array, Array3};

    #[test]
    fn test_batch_matmul() {
        // Create a batch of 2 matrices, each 2x2
        let batch_a = Array3::from_shape_vec(
            (2, 2, 2),
            vec![
                1.0, 2.0, // First matrix: [[1.0, 2.0],
                3.0, 4.0, //               [3.0, 4.0]]
                5.0, 6.0, // Second matrix: [[5.0, 6.0],
                7.0, 8.0, //                [7.0, 8.0]]
            ],
        )
        .unwrap();

        // Create a 2x2 matrix to multiply with each batch matrix
        let b = array![[1.0, 2.0], [3.0, 4.0]];

        // Perform batch multiplication
        let result = batch_matmul(&batch_a.view(), &b.view()).unwrap();

        // Expected results:
        // First result: [[1.0, 2.0], [3.0, 4.0]] × [[1.0, 2.0], [3.0, 4.0]] = [[7.0, 10.0], [15.0, 22.0]]
        // Second result: [[5.0, 6.0], [7.0, 8.0]] × [[1.0, 2.0], [3.0, 4.0]] = [[23.0, 34.0], [31.0, 46.0]]
        assert_eq!(result.shape(), &[2, 2, 2]);
        assert_relative_eq!(result[[0, 0, 0]], 7.0);
        assert_relative_eq!(result[[0, 0, 1]], 10.0);
        assert_relative_eq!(result[[0, 1, 0]], 15.0);
        assert_relative_eq!(result[[0, 1, 1]], 22.0);
        assert_relative_eq!(result[[1, 0, 0]], 23.0);
        assert_relative_eq!(result[[1, 0, 1]], 34.0);
        assert_relative_eq!(result[[1, 1, 0]], 31.0);
        assert_relative_eq!(result[[1, 1, 1]], 46.0);
    }

    #[test]
    fn test_batch_matvec() {
        // Create a batch of 2 matrices, each 2x3
        let batch_a = Array3::from_shape_vec(
            (2, 2, 3),
            vec![
                1.0, 2.0, 3.0, // First matrix: [[1.0, 2.0, 3.0],
                4.0, 5.0, 6.0, //               [4.0, 5.0, 6.0]]
                7.0, 8.0, 9.0, // Second matrix: [[7.0, 8.0, 9.0],
                10.0, 11.0, 12.0, //              [10.0, 11.0, 12.0]]
            ],
        )
        .unwrap();

        // Create a vector to multiply with each batch matrix
        let x = array![1.0, 2.0, 3.0];

        // Perform batch matrix-vector multiplication
        let result = batch_matvec(&batch_a.view(), &x.view()).unwrap();

        // Expected results:
        // First result: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] × [1.0, 2.0, 3.0] = [14.0, 32.0]
        // Second result: [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]] × [1.0, 2.0, 3.0] = [50.0, 68.0]
        assert_eq!(result.shape(), &[2, 2]);
        assert_relative_eq!(result[[0, 0]], 14.0);
        assert_relative_eq!(result[[0, 1]], 32.0);
        assert_relative_eq!(result[[1, 0]], 50.0);
        assert_relative_eq!(result[[1, 1]], 68.0);
    }

    #[test]
    fn test_batch_add_row_wise() {
        // Create a batch of 2 matrices, each 2x2
        let batch_a = Array3::from_shape_vec(
            (2, 2, 2),
            vec![
                1.0, 2.0, // First matrix: [[1.0, 2.0],
                3.0, 4.0, //               [3.0, 4.0]]
                5.0, 6.0, // Second matrix: [[5.0, 6.0],
                7.0, 8.0, //                [7.0, 8.0]]
            ],
        )
        .unwrap();

        // Create a vector to add to each row of each matrix
        let v = array![10.0, 20.0];

        // Add to each row (axis=1)
        let result = batch_add(&batch_a.view(), &v.view(), 1).unwrap();

        // Expected results:
        // First matrix: [[1.0+10.0, 2.0+20.0], [3.0+10.0, 4.0+20.0]] = [[11.0, 22.0], [13.0, 24.0]]
        // Second matrix: [[5.0+10.0, 6.0+20.0], [7.0+10.0, 8.0+20.0]] = [[15.0, 26.0], [17.0, 28.0]]
        assert_eq!(result.shape(), &[2, 2, 2]);
        assert_relative_eq!(result[[0, 0, 0]], 11.0);
        assert_relative_eq!(result[[0, 0, 1]], 22.0);
        assert_relative_eq!(result[[0, 1, 0]], 13.0);
        assert_relative_eq!(result[[0, 1, 1]], 24.0);
        assert_relative_eq!(result[[1, 0, 0]], 15.0);
        assert_relative_eq!(result[[1, 0, 1]], 26.0);
        assert_relative_eq!(result[[1, 1, 0]], 17.0);
        assert_relative_eq!(result[[1, 1, 1]], 28.0);
    }

    #[test]
    fn test_batch_add_column_wise() {
        // Create a batch of 2 matrices, each 2x2
        let batch_a = Array3::from_shape_vec(
            (2, 2, 2),
            vec![
                1.0, 2.0, // First matrix: [[1.0, 2.0],
                3.0, 4.0, //               [3.0, 4.0]]
                5.0, 6.0, // Second matrix: [[5.0, 6.0],
                7.0, 8.0, //                [7.0, 8.0]]
            ],
        )
        .unwrap();

        // Create a vector to add to each column of each matrix
        let v = array![10.0, 20.0];

        // Add to each column (axis=0)
        let result = batch_add(&batch_a.view(), &v.view(), 0).unwrap();

        // Expected results:
        // First matrix: [[1.0+10.0, 2.0+10.0], [3.0+20.0, 4.0+20.0]] = [[11.0, 12.0], [23.0, 24.0]]
        // Second matrix: [[5.0+10.0, 6.0+10.0], [7.0+20.0, 8.0+20.0]] = [[15.0, 16.0], [27.0, 28.0]]
        assert_eq!(result.shape(), &[2, 2, 2]);
        assert_relative_eq!(result[[0, 0, 0]], 11.0);
        assert_relative_eq!(result[[0, 0, 1]], 12.0);
        assert_relative_eq!(result[[0, 1, 0]], 23.0);
        assert_relative_eq!(result[[0, 1, 1]], 24.0);
        assert_relative_eq!(result[[1, 0, 0]], 15.0);
        assert_relative_eq!(result[[1, 0, 1]], 16.0);
        assert_relative_eq!(result[[1, 1, 0]], 27.0);
        assert_relative_eq!(result[[1, 1, 1]], 28.0);
    }

    #[test]
    fn test_batch_sum() {
        // Create a batch of 2 matrices, each 2x2
        let batch_a = Array3::from_shape_vec(
            (2, 2, 2),
            vec![
                1.0, 2.0, // First matrix: [[1.0, 2.0],
                3.0, 4.0, //               [3.0, 4.0]]
                5.0, 6.0, // Second matrix: [[5.0, 6.0],
                7.0, 8.0, //                [7.0, 8.0]]
            ],
        )
        .unwrap();

        // Compute the sum of all matrices in the batch
        let result = batch_sum(&batch_a.view());

        // Expected result: [[1.0, 2.0], [3.0, 4.0]] + [[5.0, 6.0], [7.0, 8.0]] = [[6.0, 8.0], [10.0, 12.0]]
        assert_eq!(result.shape(), &[2, 2]);
        assert_relative_eq!(result[[0, 0]], 6.0);
        assert_relative_eq!(result[[0, 1]], 8.0);
        assert_relative_eq!(result[[1, 0]], 10.0);
        assert_relative_eq!(result[[1, 1]], 12.0);
    }

    #[test]
    fn test_batch_matmul_dimension_error() {
        // Create a batch of 2 matrices, each 2x3
        let batch_a = Array3::from_shape_vec(
            (2, 2, 3),
            vec![
                1.0, 2.0, 3.0, // First matrix: [[1.0, 2.0, 3.0],
                4.0, 5.0, 6.0, //               [4.0, 5.0, 6.0]]
                7.0, 8.0, 9.0, // Second matrix: [[7.0, 8.0, 9.0],
                10.0, 11.0, 12.0, //              [10.0, 11.0, 12.0]]
            ],
        )
        .unwrap();

        // Create a 2x2 matrix (incompatible dimensions)
        let b = array![[1.0, 2.0], [3.0, 4.0]];

        // Batch multiplication should fail with dimension error
        let result = batch_matmul(&batch_a.view(), &b.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_matvec_dimension_error() {
        // Create a batch of 2 matrices, each 2x3
        let batch_a = Array3::from_shape_vec(
            (2, 2, 3),
            vec![
                1.0, 2.0, 3.0, // First matrix: [[1.0, 2.0, 3.0],
                4.0, 5.0, 6.0, //               [4.0, 5.0, 6.0]]
                7.0, 8.0, 9.0, // Second matrix: [[7.0, 8.0, 9.0],
                10.0, 11.0, 12.0, //              [10.0, 11.0, 12.0]]
            ],
        )
        .unwrap();

        // Create a vector with incompatible dimension
        let x = array![1.0, 2.0];

        // Batch matrix-vector multiplication should fail with dimension error
        let result = batch_matvec(&batch_a.view(), &x.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_add_dimension_error() {
        // Create a batch of 2 matrices, each 2x2
        let batch_a = Array3::from_shape_vec(
            (2, 2, 2),
            vec![
                1.0, 2.0, // First matrix: [[1.0, 2.0],
                3.0, 4.0, //               [3.0, 4.0]]
                5.0, 6.0, // Second matrix: [[5.0, 6.0],
                7.0, 8.0, //                [7.0, 8.0]]
            ],
        )
        .unwrap();

        // Create a vector with incompatible dimension
        let v = array![10.0, 20.0, 30.0];

        // Batch addition should fail with dimension error
        let result = batch_add(&batch_a.view(), &v.view(), 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_add_invalid_axis() {
        // Create a batch of 2 matrices, each 2x2
        let batch_a = Array3::from_shape_vec(
            (2, 2, 2),
            vec![
                1.0, 2.0, // First matrix: [[1.0, 2.0],
                3.0, 4.0, //               [3.0, 4.0]]
                5.0, 6.0, // Second matrix: [[5.0, 6.0],
                7.0, 8.0, //                [7.0, 8.0]]
            ],
        )
        .unwrap();

        // Create a vector
        let v = array![10.0, 20.0];

        // Batch addition with invalid axis should fail
        let result = batch_add(&batch_a.view(), &v.view(), 2);
        assert!(result.is_err());
    }
}
