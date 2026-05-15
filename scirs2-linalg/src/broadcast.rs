//! NumPy-style broadcasting for linear algebra operations on higher-dimensional arrays
//!
//! This module provides broadcasting support for operations on arrays with
//! more than 2 dimensions, following NumPy's broadcasting rules.

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array, ArrayBase, ArrayD, ArrayViewD, Data, Dimension, Ix3, IxDyn};
use scirs2_core::numeric::{Float, NumAssign};
use std::fmt::Debug;
use std::iter::Sum;

/// Compute the NumPy broadcast shape from two shapes.
///
/// Rules (right-aligned):
/// - Pad the shorter shape with leading 1s.
/// - For each pair (a, b): a==b → ok; a==1 → output b; b==1 → output a; else error.
pub fn broadcast_shapes(shape_a: &[usize], shape_b: &[usize]) -> LinalgResult<Vec<usize>> {
    let ndim = shape_a.len().max(shape_b.len());
    let mut result = vec![0usize; ndim];

    for k in 0..ndim {
        // Map k in result to dimensions counting from the right
        let i = ndim - 1 - k;
        let a = if k < shape_a.len() {
            shape_a[shape_a.len() - 1 - k]
        } else {
            1
        };
        let b = if k < shape_b.len() {
            shape_b[shape_b.len() - 1 - k]
        } else {
            1
        };
        if a == b {
            result[i] = a;
        } else if a == 1 {
            result[i] = b;
        } else if b == 1 {
            result[i] = a;
        } else {
            return Err(LinalgError::DimensionError(format!(
                "Shape mismatch for broadcasting: dimension {i} has sizes {a} and {b}"
            )));
        }
    }

    Ok(result)
}

/// Broadcast a dynamic array to a given shape, returning an owned copy.
///
/// The target shape must be broadcast-compatible with the array's shape.
/// Dimensions of size 1 in the source are "stretched" to match the target.
pub fn broadcast_to<A>(array: ArrayViewD<A>, shape: &[usize]) -> LinalgResult<ArrayD<A>>
where
    A: Float + Copy + Debug + 'static,
{
    let src_shape = array.shape();

    // Validate via broadcast_shapes (source can broadcast to target)
    let computed = broadcast_shapes(src_shape, shape)?;
    if computed != shape {
        return Err(LinalgError::DimensionError(format!(
            "Array with shape {:?} cannot be broadcast to shape {:?}",
            src_shape, shape
        )));
    }

    let ndim = shape.len();
    let total: usize = shape.iter().product();
    let mut output = ArrayD::zeros(IxDyn(shape));

    // Iterate over all output elements and map each back to a source index
    for flat_out in 0..total {
        // Decode flat output index into multi-dim output coords
        let mut out_coords = vec![0usize; ndim];
        let mut remaining = flat_out;
        for d in (0..ndim).rev() {
            out_coords[d] = remaining % shape[d];
            remaining /= shape[d];
        }

        // Map output coords to source coords (right-align, broadcast dim → 0)
        let src_ndim = src_shape.len();
        let offset = ndim - src_ndim; // leading dims not present in src
        let mut src_idx = vec![0usize; src_ndim];
        for d in 0..src_ndim {
            let src_dim = src_shape[d];
            src_idx[d] = if src_dim == 1 {
                0
            } else {
                out_coords[offset + d]
            };
        }

        output[out_coords.as_slice()] = array[src_idx.as_slice()];
    }

    Ok(output)
}

/// Broadcast a list of dynamic arrays to their common broadcast shape.
///
/// Each array is broadcast to the common shape, returning owned copies.
pub fn broadcast_arrays<A>(arrays: &[ArrayViewD<A>]) -> LinalgResult<Vec<ArrayD<A>>>
where
    A: Float + Copy + Debug + 'static,
{
    if arrays.is_empty() {
        return Ok(Vec::new());
    }

    // Compute the common broadcast shape across all arrays
    let mut common_shape = arrays[0].shape().to_vec();
    for arr in &arrays[1..] {
        common_shape = broadcast_shapes(&common_shape, arr.shape())?;
    }

    // Broadcast each array to the common shape
    arrays
        .iter()
        .map(|arr| broadcast_to(arr.view(), &common_shape))
        .collect()
}

/// Decode a flat index into multi-dim coords for a given shape.
/// Returns coords in the standard row-major order.
fn flat_to_coords(flat: usize, shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    let mut coords = vec![0usize; ndim];
    let mut remaining = flat;
    for d in (0..ndim).rev() {
        coords[d] = remaining % shape[d];
        remaining /= shape[d];
    }
    coords
}

/// Map output batch coords to input batch coords applying broadcasting rules.
/// `out_batch` is right-aligned against `in_batch_shape`.
/// Dims in `in_batch_shape` that are 1 are mapped to 0.
fn map_batch_coords(out_coords: &[usize], in_batch_shape: &[usize]) -> Vec<usize> {
    let out_len = out_coords.len();
    let in_len = in_batch_shape.len();
    let mut result = vec![0usize; in_len];
    for k in 0..in_len {
        // Right-align: out dimension corresponding to in dimension k
        let out_idx_offset = out_len.saturating_sub(in_len);
        let out_dim = out_coords[out_idx_offset + k];
        result[k] = if in_batch_shape[k] == 1 { 0 } else { out_dim };
    }
    result
}

/// Trait for broadcasting support
pub trait BroadcastExt<A> {
    /// Check if two arrays are compatible for broadcasting
    fn broadcast_compatible<D2>(&self, other: &ArrayBase<D2, impl Dimension>) -> bool
    where
        D2: Data<Elem = A>;

    /// Get the shape after broadcasting
    fn broadcastshape<D2>(&self, other: &ArrayBase<D2, impl Dimension>) -> Option<Vec<usize>>
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

    fn broadcastshape<D2>(&self, other: &ArrayBase<D2, impl Dimension>) -> Option<Vec<usize>>
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

        let mut broadcastshape = vec![0; max_ndim];

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

            broadcastshape[k] = dim1.max(dim2);
        }

        Some(broadcastshape)
    }
}

/// Broadcasting matrix multiplication for 3D arrays
///
/// This function implements NumPy-style broadcasting for matrix multiplication
/// on 3D arrays. The last two dimensions are treated as matrices, and the
/// first dimension is broadcast.
#[allow(dead_code)]
pub fn broadcast_matmul_3d<A>(
    a: &ArrayBase<impl Data<Elem = A>, Ix3>,
    b: &ArrayBase<impl Data<Elem = A>, Ix3>,
) -> LinalgResult<Array<A, Ix3>>
where
    A: Float + NumAssign + Sum + Debug + 'static,
{
    let ashape = a.shape();
    let bshape = b.shape();

    // Check matrix dimensions are compatible
    let a_cols = ashape[2];
    let b_rows = bshape[1];

    if a_cols != b_rows {
        return Err(LinalgError::DimensionError(format!(
            "Matrix dimensions don't match for multiplication: ({}, {}) x ({}, {})",
            ashape[1], a_cols, b_rows, bshape[2]
        )));
    }

    // Get the batch dimension
    let batchsize = ashape[0].max(bshape[0]);

    // Check if batch dimensions can be broadcast
    if ashape[0] != bshape[0] && ashape[0] != 1 && bshape[0] != 1 {
        return Err(LinalgError::DimensionError(
            "Batch dimensions must be compatible for broadcasting".to_string(),
        ));
    }

    // Compute output shape
    let a_rows = ashape[1];
    let b_cols = bshape[2];
    let outputshape = [batchsize, a_rows, b_cols];

    // Create output array
    let mut output = Array::zeros(outputshape);

    // Perform batched matrix multiplication
    for i in 0..batchsize {
        let a_idx = if ashape[0] == 1 { 0 } else { i };
        let b_idx = if bshape[0] == 1 { 0 } else { i };

        let a_mat = a.index_axis(scirs2_core::ndarray::Axis(0), a_idx);
        let b_mat = b.index_axis(scirs2_core::ndarray::Axis(0), b_idx);
        let mut out_mat = output.index_axis_mut(scirs2_core::ndarray::Axis(0), i);

        // Standard matrix multiplication for this batch
        scirs2_core::ndarray::linalg::general_mat_mul(
            A::one(),
            &a_mat,
            &b_mat,
            A::one(),
            &mut out_mat,
        );
    }

    Ok(output)
}

/// Broadcasting matrix multiplication for dynamic dimensional arrays
///
/// This function implements NumPy-style broadcasting for matrix multiplication
/// on arrays with arbitrary dimensions. The last two dimensions are treated
/// as matrices, and the leading dimensions are broadcast together.
#[allow(dead_code)]
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

    let ashape = a.shape();
    let bshape = b.shape();

    // Check matrix dimensions are compatible
    let a_cols = ashape[ashape.len() - 1];
    let b_rows = bshape[bshape.len() - 2];

    if a_cols != b_rows {
        return Err(LinalgError::DimensionError(format!(
            "Matrix dimensions don't match for multiplication: (..., {a_cols}) x ({b_rows}, ...)"
        )));
    }

    // Get the batch dimensions (all but the last 2)
    let a_batchshape = &ashape[..ashape.len() - 2];
    let b_batchshape = &bshape[..bshape.len() - 2];

    // Compute the broadcast shape for the batch dimensions
    let batchshape = broadcast_shapes(a_batchshape, b_batchshape)?;

    // Compute output shape
    let a_rows = ashape[ashape.len() - 2];
    let b_cols = bshape[bshape.len() - 1];
    let mut outputshape = batchshape.clone();
    outputshape.push(a_rows);
    outputshape.push(b_cols);

    // Create output array
    let mut output = Array::zeros(IxDyn(&outputshape));

    // Number of batch elements in the output
    let n_batch: usize = batchshape.iter().product::<usize>().max(1);

    // Perform batched matrix multiplication with broadcasting
    for i in 0..n_batch {
        // Decode output batch flat index into multi-dim batch coords
        let out_batch_coords = flat_to_coords(i, &batchshape);

        // Map to input batch coords via broadcasting
        let a_batch_coords = map_batch_coords(&out_batch_coords, a_batchshape);
        let b_batch_coords = map_batch_coords(&out_batch_coords, b_batchshape);

        // Extract 2D slices for this batch
        let mut a_slice = Array2::zeros((a_rows, a_cols));
        let mut b_slice = Array2::zeros((b_rows, b_cols));
        let mut out_slice = Array2::zeros((a_rows, b_cols));

        for r in 0..a_rows {
            for c in 0..a_cols {
                let mut nd_idx = a_batch_coords.clone();
                nd_idx.push(r);
                nd_idx.push(c);
                a_slice[[r, c]] = a[nd_idx.as_slice()];
            }
        }

        for r in 0..b_rows {
            for c in 0..b_cols {
                let mut nd_idx = b_batch_coords.clone();
                nd_idx.push(r);
                nd_idx.push(c);
                b_slice[[r, c]] = b[nd_idx.as_slice()];
            }
        }

        // Perform matrix multiplication
        scirs2_core::ndarray::linalg::general_mat_mul(
            A::one(),
            &a_slice.view(),
            &b_slice.view(),
            A::one(),
            &mut out_slice,
        );

        // Copy result back using the output batch coords
        for r in 0..a_rows {
            for c in 0..b_cols {
                let mut nd_idx = out_batch_coords.clone();
                nd_idx.push(r);
                nd_idx.push(c);
                output[nd_idx.as_slice()] = out_slice[[r, c]];
            }
        }
    }

    Ok(output)
}

/// Broadcasting matrix-vector multiplication for dynamic dimensional arrays
#[allow(dead_code)]
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

    let ashape = a.shape();
    let xshape = x.shape();

    // Check dimensions are compatible
    let a_cols = ashape[ashape.len() - 1];
    let x_len = xshape[xshape.len() - 1];

    if a_cols != x_len {
        return Err(LinalgError::DimensionError(format!(
            "Matrix and vector dimensions don't match: (..., {a_cols}) x ({x_len})"
        )));
    }

    // Get the batch dimensions
    let a_batchshape = &ashape[..ashape.len() - 2];
    let x_batchshape = &xshape[..xshape.len() - 1];

    // Compute the broadcast shape for the batch dimensions
    let batchshape = broadcast_shapes(a_batchshape, x_batchshape)?;

    // Compute output shape
    let a_rows = ashape[ashape.len() - 2];
    let mut outputshape = batchshape.clone();
    outputshape.push(a_rows);

    // Create output array
    let mut output = Array::zeros(IxDyn(&outputshape));

    // Number of batch elements in the output
    let n_batch: usize = batchshape.iter().product::<usize>().max(1);

    // Perform batched matrix-vector multiplication with broadcasting
    for i in 0..n_batch {
        // Decode output batch flat index into multi-dim batch coords
        let out_batch_coords = flat_to_coords(i, &batchshape);

        // Map to input batch coords via broadcasting
        let a_batch_coords = map_batch_coords(&out_batch_coords, a_batchshape);
        let x_batch_coords = map_batch_coords(&out_batch_coords, x_batchshape);

        // Extract slices for this batch
        let mut a_slice = Array2::zeros((a_rows, a_cols));
        let mut x_slice = Array1::zeros(x_len);
        let mut y_slice = Array1::zeros(a_rows);

        for r in 0..a_rows {
            for c in 0..a_cols {
                let mut nd_idx = a_batch_coords.clone();
                nd_idx.push(r);
                nd_idx.push(c);
                a_slice[[r, c]] = a[nd_idx.as_slice()];
            }
        }

        for j in 0..x_len {
            let mut nd_idx = x_batch_coords.clone();
            nd_idx.push(j);
            x_slice[j] = x[nd_idx.as_slice()];
        }

        // Perform matrix-vector multiplication
        scirs2_core::ndarray::linalg::general_mat_vec_mul(
            A::one(),
            &a_slice.view(),
            &x_slice.view(),
            A::one(),
            &mut y_slice,
        );

        // Copy result back using the output batch coords
        for j in 0..a_rows {
            let mut nd_idx = out_batch_coords.clone();
            nd_idx.push(j);
            output[nd_idx.as_slice()] = y_slice[j];
        }
    }

    Ok(output)
}

use scirs2_core::ndarray::{Array1, Array2};

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_broadcast_compatible() {
        let a = array![[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];
        let b = array![[[1.0, 2.0], [3.0, 4.0]]];

        assert!(a.broadcast_compatible(&b));

        let c = array![[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]];
        assert!(!a.broadcast_compatible(&c));
    }

    #[test]
    fn test_broadcastshape() {
        let a = array![[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];
        let b = array![[[1.0, 2.0], [3.0, 4.0]]];

        let shape = a.broadcastshape(&b).expect("Operation failed");
        assert_eq!(shape, vec![2, 2, 2]);
    }

    #[test]
    fn test_broadcast_matmul_3d() {
        // Test 3D arrays (batch of 2x2 matrices)
        let a = array![[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];
        let b = array![[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]];

        let c = broadcast_matmul_3d(&a, &b).expect("Operation failed");

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

        let c = broadcast_matmul(&a, &b).expect("Operation failed");

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

        let y = broadcast_matvec(&a, &x).expect("Operation failed");

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

        let c = broadcast_matmul_3d(&a, &b).expect("Operation failed");

        // Both batches use the same B matrix (identity)
        assert_eq!(c[[0, 0, 0]], 1.0);
        assert_eq!(c[[0, 0, 1]], 2.0);
        assert_eq!(c[[1, 0, 0]], 5.0);
        assert_eq!(c[[1, 0, 1]], 6.0);
    }

    // -------------------------------------------------------------------------
    // New tests for broadcast_shapes, broadcast_to, broadcast_arrays
    // -------------------------------------------------------------------------

    #[test]
    fn test_broadcast_shapes_basic() {
        // [3, 1] + [1, 4] -> [3, 4]
        let s = broadcast_shapes(&[3, 1], &[1, 4]).expect("should broadcast");
        assert_eq!(s, vec![3, 4]);
    }

    #[test]
    fn test_broadcast_shapes_leading_ones() {
        // [5] + [1, 5] -> [1, 5]
        let s = broadcast_shapes(&[5], &[1, 5]).expect("should broadcast");
        assert_eq!(s, vec![1, 5]);
    }

    #[test]
    fn test_broadcast_shapes_incompatible() {
        // [3] + [4] -> Err
        let result = broadcast_shapes(&[3], &[4]);
        assert!(result.is_err(), "incompatible shapes must error");
    }

    #[test]
    fn test_broadcast_to_row_to_matrix() {
        use scirs2_core::ndarray::Array;
        // Shape [1, 3] broadcast to [4, 3]: each row should be the same
        let row = Array::from_shape_vec(IxDyn(&[1, 3]), vec![1.0_f64, 2.0, 3.0]).expect("shape ok");
        let mat = broadcast_to(row.view(), &[4, 3]).expect("should broadcast");
        assert_eq!(mat.shape(), &[4, 3]);
        for i in 0..4 {
            assert_eq!(mat[[i, 0]], 1.0);
            assert_eq!(mat[[i, 1]], 2.0);
            assert_eq!(mat[[i, 2]], 3.0);
        }
    }

    #[test]
    fn test_broadcast_arrays_two_compatible() {
        use scirs2_core::ndarray::Array;
        // [3, 1] and [1, 4] -> both become [3, 4]
        let a = Array::from_shape_vec(IxDyn(&[3, 1]), vec![1.0_f64, 2.0, 3.0]).expect("shape ok");
        let b = Array::from_shape_vec(IxDyn(&[1, 4]), vec![10.0_f64, 20.0, 30.0, 40.0])
            .expect("shape ok");
        let results = broadcast_arrays(&[a.view(), b.view()]).expect("should broadcast");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].shape(), &[3, 4]);
        assert_eq!(results[1].shape(), &[3, 4]);
        // a was [1,2,3] column; after broadcast row 2 col 0 == a[2,0] == 3.0
        assert_eq!(results[0][[2, 0]], 3.0);
        assert_eq!(results[0][[2, 3]], 3.0); // same value across columns
                                             // b was [10,20,30,40] row; after broadcast col 2 any row == 30.0
        assert_eq!(results[1][[0, 2]], 30.0);
        assert_eq!(results[1][[2, 2]], 30.0);
    }

    #[test]
    fn test_broadcast_matmul_dyn_broadcasting_batch() {
        // Test batch broadcasting: a has batch [2], b has batch [1] -> output batch [2]
        let a = array![[[1.0_f64, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]].into_dyn();
        // b has batch=1, will be broadcast to match a's batch=2
        let b = array![[[1.0, 2.0], [3.0, 4.0]]].into_dyn();

        let c = broadcast_matmul(&a, &b).expect("batch broadcast matmul");
        assert_eq!(c.shape(), &[2, 2, 2]);
        // Batch 0: I * [[1,2],[3,4]] = [[1,2],[3,4]]
        assert_eq!(c[[0, 0, 0]], 1.0);
        assert_eq!(c[[0, 0, 1]], 2.0);
        assert_eq!(c[[0, 1, 0]], 3.0);
        assert_eq!(c[[0, 1, 1]], 4.0);
        // Batch 1: 2*I * [[1,2],[3,4]] = [[2,4],[6,8]]
        assert_eq!(c[[1, 0, 0]], 2.0);
        assert_eq!(c[[1, 0, 1]], 4.0);
        assert_eq!(c[[1, 1, 0]], 6.0);
        assert_eq!(c[[1, 1, 1]], 8.0);
    }

    #[test]
    fn test_broadcast_matvec_dyn_broadcasting_batch() {
        // Test batch broadcasting: a has batch [2, 1], x has batch [1, 3]
        // -> output batch [2, 3], each matrix applied to each vector
        use scirs2_core::ndarray::Array;
        // a: 2 matrices each 2×2, batch shape [2, 1]
        let a_data: Vec<f64> = vec![
            // batch [0,0]: I
            1.0, 0.0, 0.0, 1.0, // batch [1,0]: 2*I
            2.0, 0.0, 0.0, 2.0,
        ];
        let a = Array::from_shape_vec(IxDyn(&[2, 1, 2, 2]), a_data).expect("shape ok");
        // x: 3 vectors of len 2, batch shape [1, 3]
        let x_data: Vec<f64> = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let x = Array::from_shape_vec(IxDyn(&[1, 3, 2]), x_data).expect("shape ok");

        let y = broadcast_matvec(&a, &x).expect("batch broadcast matvec");
        assert_eq!(y.shape(), &[2, 3, 2]);
        // batch [0,0]: I * [1,0] = [1,0]
        assert_eq!(y[[0, 0, 0]], 1.0);
        assert_eq!(y[[0, 0, 1]], 0.0);
        // batch [0,2]: I * [1,1] = [1,1]
        assert_eq!(y[[0, 2, 0]], 1.0);
        assert_eq!(y[[0, 2, 1]], 1.0);
        // batch [1,0]: 2*I * [1,0] = [2,0]
        assert_eq!(y[[1, 0, 0]], 2.0);
        assert_eq!(y[[1, 0, 1]], 0.0);
        // batch [1,2]: 2*I * [1,1] = [2,2]
        assert_eq!(y[[1, 2, 0]], 2.0);
        assert_eq!(y[[1, 2, 1]], 2.0);
    }
}
