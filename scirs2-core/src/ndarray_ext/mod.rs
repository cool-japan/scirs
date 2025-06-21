//! Extended ndarray operations for scientific computing
//!
//! This module provides additional functionality for ndarray to support
//! the advanced array operations necessary for a complete SciPy port.
//! It implements core `NumPy`-like features that are not available in the
//! base ndarray crate.

/// Re-export essential ndarray types for convenience
pub use ndarray::{
    Array, ArrayBase, ArrayView, ArrayViewMut, Axis, Dim, Dimension, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6,
    IxDyn, OwnedRepr, ShapeBuilder, SliceInfo, ViewRepr,
};

/// Advanced indexing operations (`NumPy`-like boolean masking, fancy indexing, etc.)
pub mod indexing;

/// Statistical functions for ndarray arrays (mean, median, variance, correlation, etc.)
pub mod stats;

/// Matrix operations (eye, diag, kron, etc.)
pub mod matrix;

/// Array manipulation operations (flip, roll, tile, repeat, etc.)
pub mod manipulation;

/// Reshape a 2D array to a new shape without copying data when possible
///
/// # Arguments
///
/// * `array` - The input array
/// * `shape` - The new shape (rows, cols), which must be compatible with the original shape
///
/// # Returns
///
/// A reshaped array view of the input array if possible, or a new array otherwise
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::reshape_2d;
///
/// let a = array![[1, 2], [3, 4]];
/// let b = reshape_2d(a.view(), (4, 1)).unwrap();
/// assert_eq!(b.shape(), &[4, 1]);
/// assert_eq!(b[[0, 0]], 1);
/// assert_eq!(b[[3, 0]], 4);
/// ```
pub fn reshape_2d<T>(
    array: ArrayView<T, Ix2>,
    shape: (usize, usize),
) -> Result<Array<T, Ix2>, &'static str>
where
    T: Clone + Default,
{
    let (rows, cols) = shape;
    let total_elements = rows * cols;

    // Check if the new shape is compatible with the original shape
    if total_elements != array.len() {
        return Err("New shape dimensions must match the total number of elements");
    }

    // Create a new array with the specified shape
    let mut result = Array::<T, Ix2>::default(shape);

    // Fill the result array with elements from the input array
    let flat_iter = array.iter();
    for (i, val) in flat_iter.enumerate() {
        let r = i / cols;
        let c = i % cols;
        result[[r, c]] = val.clone();
    }

    Ok(result)
}

/// Stack 2D arrays along a given axis
///
/// # Arguments
///
/// * `arrays` - A slice of 2D arrays to stack
/// * `axis` - The axis along which to stack (0 for rows, 1 for columns)
///
/// # Returns
///
/// A new array containing the stacked arrays
///
/// # Examples
///
/// ```
/// use ndarray::{array, Axis};
/// use scirs2_core::ndarray_ext::stack_2d;
///
/// let a = array![[1, 2], [3, 4]];
/// let b = array![[5, 6], [7, 8]];
/// let c = stack_2d(&[a.view(), b.view()], 0).unwrap();
/// assert_eq!(c.shape(), &[4, 2]);
/// ```
pub fn stack_2d<T>(arrays: &[ArrayView<T, Ix2>], axis: usize) -> Result<Array<T, Ix2>, &'static str>
where
    T: Clone + Default,
{
    if arrays.is_empty() {
        return Err("No arrays provided for stacking");
    }

    // Validate that all arrays have the same shape
    let first_shape = arrays[0].shape();
    for array in arrays.iter().skip(1) {
        if array.shape() != first_shape {
            return Err("All arrays must have the same shape for stacking");
        }
    }

    let (rows, cols) = (first_shape[0], first_shape[1]);

    // Calculate the new shape
    let (new_rows, new_cols) = match axis {
        0 => (rows * arrays.len(), cols), // Stack vertically
        1 => (rows, cols * arrays.len()), // Stack horizontally
        _ => return Err("Axis must be 0 or 1 for 2D arrays"),
    };

    // Create a new array to hold the stacked result
    let mut result = Array::<T, Ix2>::default((new_rows, new_cols));

    // Copy data from the input arrays to the result
    match axis {
        0 => {
            // Stack vertically (along rows)
            for (array_idx, array) in arrays.iter().enumerate() {
                let start_row = array_idx * rows;
                for r in 0..rows {
                    for c in 0..cols {
                        result[[start_row + r, c]] = array[[r, c]].clone();
                    }
                }
            }
        }
        1 => {
            // Stack horizontally (along columns)
            for (array_idx, array) in arrays.iter().enumerate() {
                let start_col = array_idx * cols;
                for r in 0..rows {
                    for c in 0..cols {
                        result[[r, start_col + c]] = array[[r, c]].clone();
                    }
                }
            }
        }
        _ => unreachable!(),
    }

    Ok(result)
}

/// Swap axes (transpose) of a 2D array
///
/// # Arguments
///
/// * `array` - The input 2D array
///
/// # Returns
///
/// A view of the input array with the axes swapped
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::transpose_2d;
///
/// let a = array![[1, 2, 3], [4, 5, 6]];
/// let b = transpose_2d(a.view());
/// assert_eq!(b.shape(), &[3, 2]);
/// assert_eq!(b[[0, 0]], 1);
/// assert_eq!(b[[0, 1]], 4);
/// ```
pub fn transpose_2d<T>(array: ArrayView<T, Ix2>) -> Array<T, Ix2>
where
    T: Clone,
{
    array.t().to_owned()
}

/// Split a 2D array into multiple sub-arrays along a given axis
///
/// # Arguments
///
/// * `array` - The input 2D array to split
/// * `indices` - Indices where the array should be split
/// * `axis` - The axis along which to split (0 for rows, 1 for columns)
///
/// # Returns
///
/// A vector of arrays resulting from the split
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::split_2d;
///
/// let a = array![[1, 2, 3, 4], [5, 6, 7, 8]];
/// let result = split_2d(a.view(), &[2], 1).unwrap();
/// assert_eq!(result.len(), 2);
/// assert_eq!(result[0].shape(), &[2, 2]);
/// assert_eq!(result[1].shape(), &[2, 2]);
/// ```
pub fn split_2d<T>(
    array: ArrayView<T, Ix2>,
    indices: &[usize],
    axis: usize,
) -> Result<Vec<Array<T, Ix2>>, &'static str>
where
    T: Clone + Default,
{
    if indices.is_empty() {
        return Ok(vec![array.to_owned()]);
    }

    let (rows, cols) = (array.shape()[0], array.shape()[1]);
    let axis_len = if axis == 0 { rows } else { cols };

    // Validate indices
    for &idx in indices {
        if idx >= axis_len {
            return Err("Split index out of bounds");
        }
    }

    // Sort indices to ensure they're in ascending order
    let mut sorted_indices = indices.to_vec();
    sorted_indices.sort_unstable();

    // Calculate the sub-array boundaries
    let mut starts = vec![0];
    starts.extend_from_slice(&sorted_indices);

    let mut ends = sorted_indices.clone();
    ends.push(axis_len);

    // Create the split sub-arrays
    let mut result = Vec::with_capacity(starts.len());

    match axis {
        0 => {
            // Split along rows
            for (&start, &end) in starts.iter().zip(ends.iter()) {
                let sub_rows = end - start;
                let mut sub_array = Array::<T, Ix2>::default((sub_rows, cols));

                for r in 0..sub_rows {
                    for c in 0..cols {
                        sub_array[[r, c]] = array[[start + r, c]].clone();
                    }
                }

                result.push(sub_array);
            }
        }
        1 => {
            // Split along columns
            for (&start, &end) in starts.iter().zip(ends.iter()) {
                let sub_cols = end - start;
                let mut sub_array = Array::<T, Ix2>::default((rows, sub_cols));

                for r in 0..rows {
                    for c in 0..sub_cols {
                        sub_array[[r, c]] = array[[r, start + c]].clone();
                    }
                }

                result.push(sub_array);
            }
        }
        _ => return Err("Axis must be 0 or 1 for 2D arrays"),
    }

    Ok(result)
}

/// Take elements from a 2D array along a given axis using indices from another array
///
/// # Arguments
///
/// * `array` - The input 2D array
/// * `indices` - Array of indices to take
/// * `axis` - The axis along which to take values (0 for rows, 1 for columns)
///
/// # Returns
///
/// An array of values at the specified indices along the given axis
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::take_2d;
///
/// let a = array![[1, 2, 3], [4, 5, 6]];
/// let indices = array![0, 2];
/// let result = take_2d(a.view(), indices.view(), 1).unwrap();
/// assert_eq!(result.shape(), &[2, 2]);
/// assert_eq!(result[[0, 0]], 1);
/// assert_eq!(result[[0, 1]], 3);
/// ```
pub fn take_2d<T>(
    array: ArrayView<T, Ix2>,
    indices: ArrayView<usize, Ix1>,
    axis: usize,
) -> Result<Array<T, Ix2>, &'static str>
where
    T: Clone + Default,
{
    let (rows, cols) = (array.shape()[0], array.shape()[1]);
    let axis_len = if axis == 0 { rows } else { cols };

    // Check that indices are within bounds
    for &idx in indices.iter() {
        if idx >= axis_len {
            return Err("Index out of bounds");
        }
    }

    // Create the result array with the appropriate shape
    let (result_rows, result_cols) = match axis {
        0 => (indices.len(), cols),
        1 => (rows, indices.len()),
        _ => return Err("Axis must be 0 or 1 for 2D arrays"),
    };

    let mut result = Array::<T, Ix2>::default((result_rows, result_cols));

    // Fill the result array
    match axis {
        0 => {
            // Take along rows
            for (i, &idx) in indices.iter().enumerate() {
                for j in 0..cols {
                    result[[i, j]] = array[[idx, j]].clone();
                }
            }
        }
        1 => {
            // Take along columns
            for i in 0..rows {
                for (j, &idx) in indices.iter().enumerate() {
                    result[[i, j]] = array[[i, idx]].clone();
                }
            }
        }
        _ => unreachable!(),
    }

    Ok(result)
}

/// Filter an array using a boolean mask
///
/// # Arguments
///
/// * `array` - The input array
/// * `mask` - Boolean mask of the same shape as the array
///
/// # Returns
///
/// A 1D array containing the elements where the mask is true
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::mask_select;
///
/// let a = array![[1, 2, 3], [4, 5, 6]];
/// let mask = array![[true, false, true], [false, true, false]];
/// let result = mask_select(a.view(), mask.view()).unwrap();
/// assert_eq!(result.shape(), &[3]);
/// assert_eq!(result[0], 1);
/// assert_eq!(result[1], 3);
/// assert_eq!(result[2], 5);
/// ```
pub fn mask_select<T>(
    array: ArrayView<T, Ix2>,
    mask: ArrayView<bool, Ix2>,
) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Default,
{
    // Check that the mask has the same shape as the array
    if array.shape() != mask.shape() {
        return Err("Mask shape must match array shape");
    }

    // Count the number of true values in the mask
    let true_count = mask.iter().filter(|&&x| x).count();

    // Create the result array
    let mut result = Array::<T, Ix1>::default(true_count);

    // Fill the result array with elements where the mask is true
    let mut idx = 0;
    for (val, &m) in array.iter().zip(mask.iter()) {
        if m {
            result[idx] = val.clone();
            idx += 1;
        }
    }

    Ok(result)
}

/// Index a 2D array with a list of index arrays (fancy indexing)
///
/// # Arguments
///
/// * `array` - The input 2D array
/// * `row_indices` - Array of row indices
/// * `col_indices` - Array of column indices
///
/// # Returns
///
/// A 1D array containing the elements at the specified indices
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::fancy_index_2d;
///
/// let a = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
/// let row_indices = array![0, 2];
/// let col_indices = array![0, 1];
/// let result = fancy_index_2d(a.view(), row_indices.view(), col_indices.view()).unwrap();
/// assert_eq!(result.shape(), &[2]);
/// assert_eq!(result[0], 1);
/// assert_eq!(result[1], 8);
/// ```
pub fn fancy_index_2d<T>(
    array: ArrayView<T, Ix2>,
    row_indices: ArrayView<usize, Ix1>,
    col_indices: ArrayView<usize, Ix1>,
) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Default,
{
    // Check that all index arrays have the same length
    let result_size = row_indices.len();
    if col_indices.len() != result_size {
        return Err("Row and column index arrays must have the same length");
    }

    let (rows, cols) = (array.shape()[0], array.shape()[1]);

    // Check that indices are within bounds
    for &idx in row_indices.iter() {
        if idx >= rows {
            return Err("Row index out of bounds");
        }
    }

    for &idx in col_indices.iter() {
        if idx >= cols {
            return Err("Column index out of bounds");
        }
    }

    // Create the result array
    let mut result = Array::<T, Ix1>::default(result_size);

    // Fill the result array
    for i in 0..result_size {
        let row = row_indices[i];
        let col = col_indices[i];
        result[i] = array[[row, col]].clone();
    }

    Ok(result)
}

/// Select elements from an array where a condition is true
///
/// # Arguments
///
/// * `array` - The input array
/// * `condition` - A function that takes a reference to an element and returns a bool
///
/// # Returns
///
/// A 1D array containing the elements where the condition is true
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::where_condition;
///
/// let a = array![[1, 2, 3], [4, 5, 6]];
/// let result = where_condition(a.view(), |&x| x > 3).unwrap();
/// assert_eq!(result.shape(), &[3]);
/// assert_eq!(result[0], 4);
/// assert_eq!(result[1], 5);
/// assert_eq!(result[2], 6);
/// ```
pub fn where_condition<T, F>(
    array: ArrayView<T, Ix2>,
    condition: F,
) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Default,
    F: Fn(&T) -> bool,
{
    // Build a boolean mask array based on the condition
    let mask = array.map(condition);

    // Use the mask_select function to select elements
    mask_select(array, mask.view())
}

/// Check if two shapes are broadcast compatible
///
/// # Arguments
///
/// * `shape1` - First shape as a slice
/// * `shape2` - Second shape as a slice
///
/// # Returns
///
/// `true` if the shapes are broadcast compatible, `false` otherwise
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::is_broadcast_compatible;
///
/// assert!(is_broadcast_compatible(&[2, 3], &[3]));
/// // This example has dimensions that don't match (5 vs 3 in dimension 0)
/// // and aren't broadcasting compatible (neither is 1)
/// assert!(!is_broadcast_compatible(&[5, 1, 4], &[3, 1, 1]));
/// assert!(!is_broadcast_compatible(&[2, 3], &[4]));
/// ```
pub fn is_broadcast_compatible(shape1: &[usize], shape2: &[usize]) -> bool {
    // Align shapes to have the same dimensionality by prepending with 1s
    let max_dim = shape1.len().max(shape2.len());

    // Fill in 1s for missing dimensions
    let get_dim = |shape: &[usize], i: usize| -> usize {
        let offset = max_dim - shape.len();
        if i < offset {
            1 // Implicit dimension of size 1
        } else {
            shape[i - offset]
        }
    };

    // Check broadcasting rules for each dimension
    for i in 0..max_dim {
        let dim1 = get_dim(shape1, i);
        let dim2 = get_dim(shape2, i);

        // Dimensions must either be the same or one of them must be 1
        if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
            return false;
        }
    }

    true
}

/// Calculate the broadcasted shape from two input shapes
///
/// # Arguments
///
/// * `shape1` - First shape as a slice
/// * `shape2` - Second shape as a slice
///
/// # Returns
///
/// The broadcasted shape as a `Vec<usize>`, or `None` if the shapes are incompatible
pub fn broadcast_shape(shape1: &[usize], shape2: &[usize]) -> Option<Vec<usize>> {
    if !is_broadcast_compatible(shape1, shape2) {
        return None;
    }

    // Align shapes to have the same dimensionality
    let max_dim = shape1.len().max(shape2.len());
    let mut result = Vec::with_capacity(max_dim);

    // Fill in 1s for missing dimensions
    let get_dim = |shape: &[usize], i: usize| -> usize {
        let offset = max_dim - shape.len();
        if i < offset {
            1 // Implicit dimension of size 1
        } else {
            shape[i - offset]
        }
    };

    // Calculate the broadcasted shape
    for i in 0..max_dim {
        let dim1 = get_dim(shape1, i);
        let dim2 = get_dim(shape2, i);

        // The broadcasted dimension is the maximum of the two
        result.push(dim1.max(dim2));
    }

    Some(result)
}

/// Broadcast a 1D array to a 2D shape by repeating it along the specified axis
///
/// # Arguments
///
/// * `array` - The input 1D array
/// * `repeats` - Number of times to repeat the array
/// * `axis` - Axis along which to repeat (0 for rows, 1 for columns)
///
/// # Returns
///
/// A 2D array with the input array repeated along the specified axis
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::broadcast_1d_to_2d;
///
/// let a = array![1, 2, 3];
/// let b = broadcast_1d_to_2d(a.view(), 2, 0).unwrap();
/// assert_eq!(b.shape(), &[2, 3]);
/// assert_eq!(b[[0, 0]], 1);
/// assert_eq!(b[[1, 0]], 1);
/// ```
pub fn broadcast_1d_to_2d<T>(
    array: ArrayView<T, Ix1>,
    repeats: usize,
    axis: usize,
) -> Result<Array<T, Ix2>, &'static str>
where
    T: Clone + Default,
{
    let len = array.len();

    // Create the result array with the appropriate shape
    let (rows, cols) = match axis {
        0 => (repeats, len), // Broadcast along rows
        1 => (len, repeats), // Broadcast along columns
        _ => return Err("Axis must be 0 or 1"),
    };

    let mut result = Array::<T, Ix2>::default((rows, cols));

    // Fill the result array
    match axis {
        0 => {
            // Broadcast along rows
            for i in 0..repeats {
                for j in 0..len {
                    result[[i, j]] = array[j].clone();
                }
            }
        }
        1 => {
            // Broadcast along columns
            for i in 0..len {
                for j in 0..repeats {
                    result[[i, j]] = array[i].clone();
                }
            }
        }
        _ => unreachable!(),
    }

    Ok(result)
}

/// Apply an element-wise binary operation to two arrays with broadcasting
///
/// # Arguments
///
/// * `a` - First array (2D)
/// * `b` - Second array (can be 1D or 2D)
/// * `op` - Binary operation to apply to each pair of elements
///
/// # Returns
///
/// A 2D array containing the result of the operation applied element-wise
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::broadcast_apply;
///
/// let a = array![[1, 2, 3], [4, 5, 6]];
/// let b = array![10, 20, 30];
/// let result = broadcast_apply(a.view(), b.view(), |x, y| x + y).unwrap();
/// assert_eq!(result.shape(), &[2, 3]);
/// assert_eq!(result[[0, 0]], 11);
/// assert_eq!(result[[1, 2]], 36);
/// ```
pub fn broadcast_apply<T, R, F>(
    a: ArrayView<T, Ix2>,
    b: ArrayView<T, Ix1>,
    op: F,
) -> Result<Array<R, Ix2>, &'static str>
where
    T: Clone + Default,
    R: Clone + Default,
    F: Fn(&T, &T) -> R,
{
    let (a_rows, a_cols) = (a.shape()[0], a.shape()[1]);
    let b_len = b.len();

    // Check that the arrays are broadcast compatible
    if a_cols != b_len {
        return Err("Arrays are not broadcast compatible");
    }

    // Create the result array
    let mut result = Array::<R, Ix2>::default((a_rows, a_cols));

    // Apply the operation element-wise
    for i in 0..a_rows {
        for j in 0..a_cols {
            result[[i, j]] = op(&a[[i, j]], &b[j]);
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_reshape_2d() {
        let a = array![[1, 2], [3, 4]];
        let b = reshape_2d(a.view(), (4, 1)).unwrap();
        assert_eq!(b.shape(), &[4, 1]);
        assert_eq!(b[[0, 0]], 1);
        assert_eq!(b[[1, 0]], 2);
        assert_eq!(b[[2, 0]], 3);
        assert_eq!(b[[3, 0]], 4);

        // Test invalid shape
        let result = reshape_2d(a.view(), (3, 1));
        assert!(result.is_err());
    }

    #[test]
    fn test_stack_2d() {
        let a = array![[1, 2], [3, 4]];
        let b = array![[5, 6], [7, 8]];

        // Stack vertically (along axis 0)
        let c = stack_2d(&[a.view(), b.view()], 0).unwrap();
        assert_eq!(c.shape(), &[4, 2]);
        assert_eq!(c[[0, 0]], 1);
        assert_eq!(c[[1, 0]], 3);
        assert_eq!(c[[2, 0]], 5);
        assert_eq!(c[[3, 0]], 7);

        // Stack horizontally (along axis 1)
        let d = stack_2d(&[a.view(), b.view()], 1).unwrap();
        assert_eq!(d.shape(), &[2, 4]);
        assert_eq!(d[[0, 0]], 1);
        assert_eq!(d[[0, 1]], 2);
        assert_eq!(d[[0, 2]], 5);
        assert_eq!(d[[0, 3]], 6);
    }

    #[test]
    fn test_transpose_2d() {
        let a = array![[1, 2, 3], [4, 5, 6]];
        let b = transpose_2d(a.view());
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b[[0, 0]], 1);
        assert_eq!(b[[0, 1]], 4);
        assert_eq!(b[[1, 0]], 2);
        assert_eq!(b[[2, 1]], 6);
    }

    #[test]
    fn test_split_2d() {
        let a = array![[1, 2, 3, 4], [5, 6, 7, 8]];

        // Split along columns at index 2
        let result = split_2d(a.view(), &[2], 1).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].shape(), &[2, 2]);
        assert_eq!(result[0][[0, 0]], 1);
        assert_eq!(result[0][[0, 1]], 2);
        assert_eq!(result[0][[1, 0]], 5);
        assert_eq!(result[0][[1, 1]], 6);
        assert_eq!(result[1].shape(), &[2, 2]);
        assert_eq!(result[1][[0, 0]], 3);
        assert_eq!(result[1][[0, 1]], 4);
        assert_eq!(result[1][[1, 0]], 7);
        assert_eq!(result[1][[1, 1]], 8);

        // Split along rows at index 1
        let result = split_2d(a.view(), &[1], 0).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].shape(), &[1, 4]);
        assert_eq!(result[1].shape(), &[1, 4]);
    }

    #[test]
    fn test_take_2d() {
        let a = array![[1, 2, 3], [4, 5, 6]];
        let indices = array![0, 2];

        // Take along columns
        let result = take_2d(a.view(), indices.view(), 1).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result[[0, 0]], 1);
        assert_eq!(result[[0, 1]], 3);
        assert_eq!(result[[1, 0]], 4);
        assert_eq!(result[[1, 1]], 6);
    }

    #[test]
    fn test_mask_select() {
        let a = array![[1, 2, 3], [4, 5, 6]];
        let mask = array![[true, false, true], [false, true, false]];

        let result = mask_select(a.view(), mask.view()).unwrap();
        assert_eq!(result.shape(), &[3]);
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 3);
        assert_eq!(result[2], 5);
    }

    #[test]
    fn test_fancy_index_2d() {
        let a = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        let row_indices = array![0, 2];
        let col_indices = array![0, 1];

        let result = fancy_index_2d(a.view(), row_indices.view(), col_indices.view()).unwrap();
        assert_eq!(result.shape(), &[2]);
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 8);
    }

    #[test]
    fn test_where_condition() {
        let a = array![[1, 2, 3], [4, 5, 6]];
        let result = where_condition(a.view(), |&x| x > 3).unwrap();
        assert_eq!(result.shape(), &[3]);
        assert_eq!(result[0], 4);
        assert_eq!(result[1], 5);
        assert_eq!(result[2], 6);
    }

    #[test]
    fn test_broadcast_1d_to_2d() {
        let a = array![1, 2, 3];

        // Broadcast along rows (axis 0)
        let b = broadcast_1d_to_2d(a.view(), 2, 0).unwrap();
        assert_eq!(b.shape(), &[2, 3]);
        assert_eq!(b[[0, 0]], 1);
        assert_eq!(b[[0, 1]], 2);
        assert_eq!(b[[1, 0]], 1);
        assert_eq!(b[[1, 2]], 3);

        // Broadcast along columns (axis 1)
        let c = broadcast_1d_to_2d(a.view(), 2, 1).unwrap();
        assert_eq!(c.shape(), &[3, 2]);
        assert_eq!(c[[0, 0]], 1);
        assert_eq!(c[[0, 1]], 1);
        assert_eq!(c[[1, 0]], 2);
        assert_eq!(c[[2, 1]], 3);
    }

    #[test]
    fn test_broadcast_apply() {
        let a = array![[1, 2, 3], [4, 5, 6]];
        let b = array![10, 20, 30];

        let result = broadcast_apply(a.view(), b.view(), |x, y| x + y).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result[[0, 0]], 11);
        assert_eq!(result[[0, 1]], 22);
        assert_eq!(result[[0, 2]], 33);
        assert_eq!(result[[1, 0]], 14);
        assert_eq!(result[[1, 1]], 25);
        assert_eq!(result[[1, 2]], 36);

        let result = broadcast_apply(a.view(), b.view(), |x, y| x * y).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result[[0, 0]], 10);
        assert_eq!(result[[0, 1]], 40);
        assert_eq!(result[[0, 2]], 90);
        assert_eq!(result[[1, 0]], 40);
        assert_eq!(result[[1, 1]], 100);
        assert_eq!(result[[1, 2]], 180);
    }
}
