//! NumPy-style broadcasting for linear algebra operations on higher-dimensional arrays
//!
//! This module provides broadcasting support for operations on arrays with
//! more than 2 dimensions, following NumPy's broadcasting rules.

use crate::error::{LinalgError, LinalgResult};
use ndarray::{Array, ArrayBase, Data, Dimension, Ix3, IxDyn};
use num_traits::{Float, NumAssign};
use std::fmt::Debug;
use std::iter::Sum;

/// Trait for broadcasting support
pub trait BroadcastExt<A> {
    /// Check if two arrays are compatible for broadcasting
    fn broadcast_compatible<D2>(&self, other: &ArrayBase<D2, impl Dimension>) -> bool
    where
        D2: Data<Elem = A>;

    /// Get the shape after broadcasting
    fn broadcast_shape<D2>(&self, other: &ArrayBase<D2, impl Dimension>) -> Option<Vec<usize>>
    where
        D2: Data<Elem = A>;
}

impl<A, S, D> BroadcastExt<A> for ArrayBase<S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    fn broadcast_compatible<D2>(&self, other: &ArrayBase<D2, impl Dimension>) -> bool
    where
        D2: Data<Elem = A>,
    {
        let shape1 = self.shape();
        let shape2 = other.shape();
        let ndim1 = shape1.len();
        let ndim2 = shape2.len();

        // Start from the trailing dimensions
        let mut i = ndim1;
        let mut j = ndim2;

        while i > 0 && j > 0 {
            i -= 1;
            j -= 1;

            let dim1 = shape1[i];
            let dim2 = shape2[j];

            // Dimensions are compatible if they are equal or one of them is 1
            if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
                return false;
            }
        }

        true
    }

    fn broadcast_shape<D2>(&self, other: &ArrayBase<D2, impl Dimension>) -> Option<Vec<usize>>
    where
        D2: Data<Elem = A>,
    {
        if !self.broadcast_compatible(other) {
            return None;
        }

        let shape1 = self.shape();
        let shape2 = other.shape();
        let ndim1 = shape1.len();
        let ndim2 = shape2.len();
        let max_ndim = ndim1.max(ndim2);

        let mut broadcast_shape = vec![0; max_ndim];

        // Fill from the trailing dimensions
        let mut i = ndim1;
        let mut j = ndim2;
        let mut k = max_ndim;

        while k > 0 {
            k -= 1;

            let dim1 = if i > 0 {
                i -= 1;
                shape1[i]
            } else {
                1
            };

            let dim2 = if j > 0 {
                j -= 1;
                shape2[j]
            } else {
                1
            };

            broadcast_shape[k] = dim1.max(dim2);
        }

        Some(broadcast_shape)
    }
}

/// Broadcasting matrix multiplication for 3D arrays
///
/// This function implements NumPy-style broadcasting for matrix multiplication
/// on 3D arrays. The last two dimensions are treated as matrices, and the
/// first dimension is broadcast.
pub fn broadcast_matmul_3d<A>(
    a: &ArrayBase<impl Data<Elem = A>, Ix3>,
    b: &ArrayBase<impl Data<Elem = A>, Ix3>,
) -> LinalgResult<Array<A, Ix3>>
where
    A: Float + NumAssign + Sum + Debug + 'static,
{
    let a_shape = a.shape();
    let b_shape = b.shape();

    // Check matrix dimensions are compatible
    let a_cols = a_shape[2];
    let b_rows = b_shape[1];

    if a_cols != b_rows {
        return Err(LinalgError::DimensionError(format!(
            "Matrix dimensions don't match for multiplication: ({}, {}) x ({}, {})",
            a_shape[1], a_cols, b_rows, b_shape[2]
        )));
    }

    // Get the batch dimension
    let batch_size = a_shape[0].max(b_shape[0]);

    // Check if batch dimensions can be broadcast
    if a_shape[0] != b_shape[0] && a_shape[0] != 1 && b_shape[0] != 1 {
        return Err(LinalgError::DimensionError(
            "Batch dimensions must be compatible for broadcasting".to_string(),
        ));
    }

    // Compute output shape
    let a_rows = a_shape[1];
    let b_cols = b_shape[2];
    let output_shape = [batch_size, a_rows, b_cols];

    // Create output array
    let mut output = Array::zeros(output_shape);

    // Perform batched matrix multiplication
    for i in 0..batch_size {
        let a_idx = if a_shape[0] == 1 { 0 } else { i };
        let b_idx = if b_shape[0] == 1 { 0 } else { i };

        let a_mat = a.index_axis(ndarray::Axis(0), a_idx);
        let b_mat = b.index_axis(ndarray::Axis(0), b_idx);
        let mut out_mat = output.index_axis_mut(ndarray::Axis(0), i);

        // Standard matrix multiplication for this batch
        ndarray::linalg::general_mat_mul(A::one(), &a_mat, &b_mat, A::one(), &mut out_mat);
    }

    Ok(output)
}

/// Broadcasting matrix multiplication for dynamic dimensional arrays
///
/// This function implements NumPy-style broadcasting for matrix multiplication
/// on arrays with arbitrary dimensions. The last two dimensions are treated
/// as matrices, and the leading dimensions are broadcast together.
pub fn broadcast_matmul<A>(
    a: &ArrayBase<impl Data<Elem = A>, IxDyn>,
    b: &ArrayBase<impl Data<Elem = A>, IxDyn>,
) -> LinalgResult<Array<A, IxDyn>>
where
    A: Float + NumAssign + Sum + Debug + 'static,
{
    // Check that arrays have at least 2 dimensions
    if a.ndim() < 2 || b.ndim() < 2 {
        return Err(LinalgError::DimensionError(
            "Arrays must have at least 2 dimensions for matrix multiplication".to_string(),
        ));
    }

    let a_shape = a.shape();
    let b_shape = b.shape();

    // Check matrix dimensions are compatible
    let a_cols = a_shape[a_shape.len() - 1];
    let b_rows = b_shape[b_shape.len() - 2];

    if a_cols != b_rows {
        return Err(LinalgError::DimensionError(format!(
            "Matrix dimensions don't match for multiplication: (..., {}) x ({}, ...)",
            a_cols, b_rows
        )));
    }

    // Get the batch dimensions (all but the last 2)
    let a_batch_shape = &a_shape[..a_shape.len() - 2];
    let b_batch_shape = &b_shape[..b_shape.len() - 2];

    // Check if batch dimensions can be broadcast
    let batch_shape = if a_batch_shape == b_batch_shape {
        a_batch_shape.to_vec()
    } else {
        // For now, we don't support full broadcasting - require exact match
        return Err(LinalgError::DimensionError(
            "Batch dimensions must match exactly (full broadcasting not yet implemented)"
                .to_string(),
        ));
    };

    // Compute output shape
    let a_rows = a_shape[a_shape.len() - 2];
    let b_cols = b_shape[b_shape.len() - 1];
    let mut output_shape = batch_shape;
    output_shape.push(a_rows);
    output_shape.push(b_cols);

    // Create output array
    let mut output = Array::zeros(IxDyn(&output_shape));

    // Extract the matrix dimensions
    let n_batch = output.len() / (a_rows * b_cols);

    // Perform batched matrix multiplication
    // Need to reshape in steps to avoid borrowing issues
    for i in 0..n_batch {
        // Extract 2D slices for this batch
        let mut a_slice = Array2::zeros((a_rows, a_cols));
        let mut b_slice = Array2::zeros((b_rows, b_cols));
        let mut out_slice = Array2::zeros((a_rows, b_cols));

        // Copy data into slices
        let a_start = i * a_rows * a_cols;
        let b_start = i * b_rows * b_cols;
        let out_start = i * a_rows * b_cols;

        for r in 0..a_rows {
            for c in 0..a_cols {
                let flat_idx = a_start + r * a_cols + c;
                let nd_idx: Vec<usize> = {
                    let mut idx = vec![0; a.ndim()];
                    let mut remaining = flat_idx;
                    for dim in (0..a.ndim()).rev() {
                        idx[dim] = remaining % a_shape[dim];
                        remaining /= a_shape[dim];
                    }
                    idx
                };
                a_slice[[r, c]] = a[nd_idx.as_slice()];
            }
        }

        for r in 0..b_rows {
            for c in 0..b_cols {
                let flat_idx = b_start + r * b_cols + c;
                let nd_idx: Vec<usize> = {
                    let mut idx = vec![0; b.ndim()];
                    let mut remaining = flat_idx;
                    for dim in (0..b.ndim()).rev() {
                        idx[dim] = remaining % b_shape[dim];
                        remaining /= b_shape[dim];
                    }
                    idx
                };
                b_slice[[r, c]] = b[nd_idx.as_slice()];
            }
        }

        // Perform matrix multiplication
        ndarray::linalg::general_mat_mul(
            A::one(),
            &a_slice.view(),
            &b_slice.view(),
            A::one(),
            &mut out_slice,
        );

        // Copy result back
        for r in 0..a_rows {
            for c in 0..b_cols {
                let flat_idx = out_start + r * b_cols + c;
                let nd_idx: Vec<usize> = {
                    let mut idx = vec![0; output.ndim()];
                    let mut remaining = flat_idx;
                    for dim in (0..output.ndim()).rev() {
                        idx[dim] = remaining % output_shape[dim];
                        remaining /= output_shape[dim];
                    }
                    idx
                };
                output[nd_idx.as_slice()] = out_slice[[r, c]];
            }
        }
    }

    Ok(output)
}

/// Broadcasting matrix-vector multiplication for dynamic dimensional arrays
pub fn broadcast_matvec<A>(
    a: &ArrayBase<impl Data<Elem = A>, IxDyn>,
    x: &ArrayBase<impl Data<Elem = A>, IxDyn>,
) -> LinalgResult<Array<A, IxDyn>>
where
    A: Float + NumAssign + Sum + Debug + 'static,
{
    // Check that matrix has at least 2 dimensions and vector has at least 1
    if a.ndim() < 2 || x.ndim() < 1 {
        return Err(LinalgError::DimensionError(
            "Matrix must have at least 2 dimensions and vector at least 1".to_string(),
        ));
    }

    let a_shape = a.shape();
    let x_shape = x.shape();

    // Check dimensions are compatible
    let a_cols = a_shape[a_shape.len() - 1];
    let x_len = x_shape[x_shape.len() - 1];

    if a_cols != x_len {
        return Err(LinalgError::DimensionError(format!(
            "Matrix and vector dimensions don't match: (..., {}) x ({})",
            a_cols, x_len
        )));
    }

    // Get the batch dimensions
    let a_batch_shape = &a_shape[..a_shape.len() - 2];
    let x_batch_shape = &x_shape[..x_shape.len() - 1];

    // Check if batch dimensions can be broadcast
    let batch_shape = if a_batch_shape == x_batch_shape {
        a_batch_shape.to_vec()
    } else {
        // For now, we don't support full broadcasting
        return Err(LinalgError::DimensionError(
            "Batch dimensions must match exactly (full broadcasting not yet implemented)"
                .to_string(),
        ));
    };

    // Compute output shape
    let a_rows = a_shape[a_shape.len() - 2];
    let mut output_shape = batch_shape;
    output_shape.push(a_rows);

    // Create output array
    let mut output = Array::zeros(IxDyn(&output_shape));

    // Extract dimensions
    let n_batch = output.len() / a_rows;

    // Perform batched matrix-vector multiplication
    for i in 0..n_batch {
        // Extract slices for this batch
        let mut a_slice = Array2::zeros((a_rows, a_cols));
        let mut x_slice = Array1::zeros(x_len);
        let mut y_slice = Array1::zeros(a_rows);

        // Copy data into slices
        let a_start = i * a_rows * a_cols;
        let x_start = i * x_len;
        let y_start = i * a_rows;

        for r in 0..a_rows {
            for c in 0..a_cols {
                let flat_idx = a_start + r * a_cols + c;
                let nd_idx: Vec<usize> = {
                    let mut idx = vec![0; a.ndim()];
                    let mut remaining = flat_idx;
                    for dim in (0..a.ndim()).rev() {
                        idx[dim] = remaining % a_shape[dim];
                        remaining /= a_shape[dim];
                    }
                    idx
                };
                a_slice[[r, c]] = a[nd_idx.as_slice()];
            }
        }

        for j in 0..x_len {
            let flat_idx = x_start + j;
            let nd_idx: Vec<usize> = {
                let mut idx = vec![0; x.ndim()];
                let mut remaining = flat_idx;
                for dim in (0..x.ndim()).rev() {
                    idx[dim] = remaining % x_shape[dim];
                    remaining /= x_shape[dim];
                }
                idx
            };
            x_slice[j] = x[nd_idx.as_slice()];
        }

        // Perform matrix-vector multiplication
        ndarray::linalg::general_mat_vec_mul(
            A::one(),
            &a_slice.view(),
            &x_slice.view(),
            A::one(),
            &mut y_slice,
        );

        // Copy result back
        for j in 0..a_rows {
            let flat_idx = y_start + j;
            let nd_idx: Vec<usize> = {
                let mut idx = vec![0; output.ndim()];
                let mut remaining = flat_idx;
                for dim in (0..output.ndim()).rev() {
                    idx[dim] = remaining % output_shape[dim];
                    remaining /= output_shape[dim];
                }
                idx
            };
            output[nd_idx.as_slice()] = y_slice[j];
        }
    }

    Ok(output)
}

use ndarray::{Array1, Array2};

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_broadcast_compatible() {
        let a = array![[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];
        let b = array![[[1.0, 2.0], [3.0, 4.0]]];

        assert!(a.broadcast_compatible(&b));

        let c = array![[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]];
        assert!(!a.broadcast_compatible(&c));
    }

    #[test]
    fn test_broadcast_shape() {
        let a = array![[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];
        let b = array![[[1.0, 2.0], [3.0, 4.0]]];

        let shape = a.broadcast_shape(&b).unwrap();
        assert_eq!(shape, vec![2, 2, 2]);
    }

    #[test]
    fn test_broadcast_matmul_3d() {
        // Test 3D arrays (batch of 2x2 matrices)
        let a = array![[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];
        let b = array![[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]];

        let c = broadcast_matmul_3d(&a, &b).unwrap();

        // First batch: identity matrix multiplication
        assert_eq!(c[[0, 0, 0]], 1.0);
        assert_eq!(c[[0, 0, 1]], 2.0);
        assert_eq!(c[[0, 1, 0]], 3.0);
        assert_eq!(c[[0, 1, 1]], 4.0);

        // Second batch: multiplication by 2*I
        assert_eq!(c[[1, 0, 0]], 10.0);
        assert_eq!(c[[1, 0, 1]], 12.0);
        assert_eq!(c[[1, 1, 0]], 14.0);
        assert_eq!(c[[1, 1, 1]], 16.0);
    }

    #[test]
    fn test_broadcast_matmul_dyn() {
        // Test dynamic arrays (batch of 2x2 matrices)
        let a = array![[[1.0_f64, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]].into_dyn();
        let b = array![[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]].into_dyn();

        let c = broadcast_matmul(&a, &b).unwrap();

        // First batch: identity matrix multiplication
        assert_eq!(c[[0, 0, 0]], 1.0);
        assert_eq!(c[[0, 0, 1]], 2.0);
        assert_eq!(c[[0, 1, 0]], 3.0);
        assert_eq!(c[[0, 1, 1]], 4.0);

        // Second batch: multiplication by 2*I
        assert_eq!(c[[1, 0, 0]], 10.0);
        assert_eq!(c[[1, 0, 1]], 12.0);
        assert_eq!(c[[1, 1, 0]], 14.0);
        assert_eq!(c[[1, 1, 1]], 16.0);
    }

    #[test]
    fn test_broadcast_matvec_dyn() {
        // Test dynamic array (batch of 2x2 matrices) with dynamic vector (batch of vectors)
        let a = array![[[1.0_f64, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]].into_dyn();
        let x = array![[1.0, 1.0], [2.0, 1.0]].into_dyn();

        let y = broadcast_matvec(&a, &x).unwrap();

        // First batch: [1,2;3,4] * [1,1] = [3,7]
        assert_eq!(y[[0, 0]], 3.0);
        assert_eq!(y[[0, 1]], 7.0);

        // Second batch: [5,6;7,8] * [2,1] = [16,22]
        assert_eq!(y[[1, 0]], 16.0);
        assert_eq!(y[[1, 1]], 22.0);
    }

    #[test]
    fn test_incompatible_dimensions() {
        // These matrices have incompatible dimensions: (2, 2) x (3, 2)
        let a = array![[[1.0_f64, 2.0], [3.0, 4.0]]].into_dyn();
        let b = array![[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]].into_dyn();

        let result = broadcast_matmul(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_broadcast_3d_with_different_batch() {
        // Test broadcasting with different batch sizes (1 and 2)
        let a = array![[[1.0_f64, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];
        let b = array![[[1.0, 0.0], [0.0, 1.0]]];

        let c = broadcast_matmul_3d(&a, &b).unwrap();

        // Both batches use the same B matrix (identity)
        assert_eq!(c[[0, 0, 0]], 1.0);
        assert_eq!(c[[0, 0, 1]], 2.0);
        assert_eq!(c[[1, 0, 0]], 5.0);
        assert_eq!(c[[1, 0, 1]], 6.0);
    }
}
