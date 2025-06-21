//! Core array operations for scientific computing
//!
//! This module provides implementations of fundamental array operations
//! that correspond to `NumPy`'s core functionality, focusing on reshape,
//! transpose, concatenate, split, and other basic array manipulations.

use ndarray::{Array, ArrayView, Axis, Dimension, ShapeError};

/// Reshape an array to a new shape without copying data when possible
///
/// # Arguments
///
/// * `array` - The input array to reshape
/// * `shape` - The new shape, which must be compatible with the original shape
///
/// # Returns
///
/// A reshaped array view of the input array if possible, or a new array otherwise
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array2};
/// use scirs2_core::ndarray_ext::reshape;
///
/// let a = array![[1, 2], [3, 4]];
/// let b = reshape(a.view(), (4, 1)).unwrap();
/// assert_eq!(b.shape(), &[4, 1]);
/// assert_eq!(b[[0, 0]], 1);
/// assert_eq!(b[[3, 0]], 4);
/// ```
pub fn reshape<D1, D2, T>(array: ArrayView<T, D1>, shape: D2) -> Result<Array<T, D2::Dim>, &'static str>
where
    D1: Dimension,
    D2: ndarray::ShapeBuilder,
    T: Clone + Default,
{
    // Check if the new shape is compatible with the original shape
    let dim = shape.into_shape();
    let total_elements = dim.size();
    if total_elements != array.len() {
        return Err("New shape dimensions must match the total number of elements");
    }

    // Create a new array with the specified shape
    match Array::from_shape_vec(dim, array.iter().cloned().collect()) {
        Ok(reshaped) => Ok(reshaped),
        Err(_) => Err("Failed to reshape array"),
    }
}

/// Stack arrays along a given axis
///
/// # Arguments
///
/// * `arrays` - A slice of arrays to stack
/// * `axis` - The axis along which to stack
///
/// # Returns
///
/// A new array containing the stacked arrays
///
/// # Examples
///
/// ```
/// use ndarray::{array, Axis};
/// use scirs2_core::ndarray_ext::stack;
///
/// let a = array![[1, 2], [3, 4]];
/// let b = array![[5, 6], [7, 8]];
/// let c = stack(&[a.view(), b.view()], Axis(0)).unwrap();
/// assert_eq!(c.shape(), &[4, 2]);
/// ```
pub fn stack<D, T>(arrays: &[ArrayView<T, D>], axis: Axis) -> Result<Array<T, D>, &'static str>
where
    D: Dimension,
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

    // Calculate the new shape
    let mut new_shape = arrays[0].raw_dim();
    let axis_idx = axis.index();

    if axis_idx >= new_shape.ndim() {
        return Err("Axis index out of bounds");
    }

    // Update the size of the specified axis
    new_shape[axis_idx] = new_shape[axis_idx] * arrays.len();

    // Create a new array to hold the stacked result
    let mut result = Array::default(new_shape);

    // Copy data from the input arrays to the result
    let axis_stride = arrays[0].len_of(axis);

    // This simplified implementation only supports 2D arrays
    if arrays[0].ndim() != 2 {
        return Err("This simplified implementation only supports 2D arrays");
    }

    for (i, array) in arrays.iter().enumerate() {
        let start = i * axis_stride;

        if axis_idx == 0 {
            for j in 0..axis_stride {
                for k in 0..arrays[0].shape()[1] {
                    result[[start + j, k]] = array[[j, k]].clone();
                }
            }
        } else if axis_idx == 1 {
            for j in 0..arrays[0].shape()[0] {
                for k in 0..axis_stride {
                    result[[j, start + k]] = array[[j, k]].clone();
                }
            }
        } else {
            return Err("Only axes 0 and 1 are supported in this implementation");
        }
    }

    Ok(result)
}

/// Swap axes (transpose) of an array
///
/// # Arguments
///
/// * `array` - The input array
/// * `axis1` - First axis to swap
/// * `axis2` - Second axis to swap
///
/// # Returns
///
/// A view of the input array with the specified axes swapped
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::swapaxes;
///
/// let a = array![[1, 2, 3], [4, 5, 6]];
/// let b = swapaxes(a.view(), 0, 1).unwrap();
/// assert_eq!(b.shape(), &[3, 2]);
/// assert_eq!(b[[0, 0]], 1);
/// assert_eq!(b[[0, 1]], 4);
/// ```
pub fn swapaxes<D, T>(array: ArrayView<T, D>, axis1: usize, axis2: usize) -> Result<Array<T, D>, &'static str>
where
    D: Dimension,
    T: Clone,
{
    if axis1 >= array.ndim() || axis2 >= array.ndim() {
        return Err("Axis indices out of bounds");
    }

    // Create a new permutation of axes
    let mut permutation: Vec<usize> = (0..array.ndim()).collect();
    permutation.swap(axis1, axis2);

    // Apply the permutation using ndarray's permuted method
    // This creates a view with permuted dimensions
    let transposed_view = array.permuted_axes(permutation);

    // Convert to owned array for consistency with other functions
    Ok(transposed_view.to_owned())
}

/// Split an array into multiple sub-arrays along a given axis
///
/// # Arguments
///
/// * `array` - The input array to split
/// * `indices` - Indices where the array should be split
/// * `axis` - The axis along which to split
///
/// # Returns
///
/// A vector of arrays resulting from the split
///
/// # Examples
///
/// ```
/// use ndarray::{array, Axis};
/// use scirs2_core::ndarray_ext::split;
///
/// let a = array![[1, 2, 3, 4], [5, 6, 7, 8]];
/// let result = split(a.view(), &[2], Axis(1)).unwrap();
/// assert_eq!(result.len(), 2);
/// assert_eq!(result[0].shape(), &[2, 2]);
/// assert_eq!(result[1].shape(), &[2, 2]);
/// ```
pub fn split<D, T>(
    array: ArrayView<T, D>,
    indices: &[usize],
    axis: Axis,
) -> Result<Vec<Array<T, D>>, &'static str>
where
    D: Dimension,
    T: Clone,
{
    if indices.is_empty() {
        return Ok(vec![array.to_owned()]);
    }

    let axis_idx = axis.index();
    if axis_idx >= array.ndim() {
        return Err("Axis index out of bounds");
    }

    let axis_len = array.len_of(axis);
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

    for (&start, &end) in starts.iter().zip(ends.iter()) {
        // Create a slice specification for the current sub-array
        let mut slice_spec = vec![ndarray::s![..]; array.ndim()];
        slice_spec[axis_idx] = ndarray::s![start..end];

        // Extract the sub-array by slicing the original array
        let sub_array = array.slice(slice_spec.as_slice());
        result.push(sub_array.to_owned());
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn test_reshape() {
        let a = array![[1, 2], [3, 4]];
        let b = reshape(a.view(), (4, 1)).unwrap();
        assert_eq!(b.shape(), &[4, 1]);
        assert_eq!(b[[0, 0]], 1);
        assert_eq!(b[[1, 0]], 2);
        assert_eq!(b[[2, 0]], 3);
        assert_eq!(b[[3, 0]], 4);

        // Test invalid reshape
        let result = reshape(a.view(), (3, 1));
        assert!(result.is_err());
    }

    #[test]
    fn test_stack() {
        let a = array![[1, 2], [3, 4]];
        let b = array![[5, 6], [7, 8]];

        // Stack along axis 0
        let c = stack(&[a.view(), b.view()], Axis(0)).unwrap();
        assert_eq!(c.shape(), &[4, 2]);
        assert_eq!(c[[0, 0]], 1);
        assert_eq!(c[[2, 1]], 6);

        // Stack along axis 1
        let d = stack(&[a.view(), b.view()], Axis(1)).unwrap();
        assert_eq!(d.shape(), &[2, 4]);
        assert_eq!(d[[0, 0]], 1);
        assert_eq!(d[[0, 3]], 6);
    }

    #[test]
    fn test_swapaxes() {
        let a = array![[1, 2, 3], [4, 5, 6]];
        let b = swapaxes(a.view(), 0, 1).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b[[0, 0]], 1);
        assert_eq!(b[[0, 1]], 4);
        assert_eq!(b[[2, 0]], 3);
        assert_eq!(b[[2, 1]], 6);
    }

    #[test]
    fn test_split() {
        let a = array![[1, 2, 3, 4], [5, 6, 7, 8]];

        // Split along axis 1 at index 2
        let result = split(a.view(), &[2], Axis(1)).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].shape(), &[2, 2]);
        assert_eq!(result[0][[0, 0]], 1);
        assert_eq!(result[0][[1, 1]], 6);
        assert_eq!(result[1].shape(), &[2, 2]);
        assert_eq!(result[1][[0, 0]], 3);
        assert_eq!(result[1][[1, 1]], 8);

        // Split along axis 0 at index 1
        let result = split(a.view(), &[1], Axis(0)).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].shape(), &[1, 4]);
        assert_eq!(result[1].shape(), &[1, 4]);
    }
}