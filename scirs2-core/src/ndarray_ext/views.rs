//! Extended view operations for ndarray
//!
//! This module provides enhanced array view operations that support
//! more advanced memory layout options and zero-copy transformations.

use ndarray::{Array, ArrayView, Dimension};

/// Memory layout ordering types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order {
    /// C-style row-major order (default in ndarray)
    C,
    /// Fortran-style column-major order
    F,
}

/// Create a strided view of an array
///
/// # Arguments
///
/// * `array` - The input array
/// * `step` - Step size for each dimension
///
/// # Returns
///
/// A view of the array with the specified stride
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::strided_view;
///
/// let a = array![[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]];
/// let view = strided_view(a.view(), &[2, 2]).unwrap();
/// assert_eq!(view.shape(), &[2, 2]);
/// assert_eq!(view[[0, 0]], 1);
/// assert_eq!(view[[0, 1]], 3);
/// assert_eq!(view[[1, 0]], 9);
/// assert_eq!(view[[1, 1]], 11);
/// ```
pub fn strided_view<D, T>(
    array: ArrayView<T, D>,
    step: &[usize],
) -> Result<Array<T, D>, &'static str>
where
    D: Dimension,
    T: Clone + Default,
{
    if step.len() != array.ndim() {
        return Err("Step size array must match array dimensionality");
    }

    for (i, &s) in step.iter().enumerate() {
        if s == 0 {
            return Err("Step size must be at least 1");
        }
        if s > array.shape()[i] {
            return Err("Step size cannot exceed dimension size");
        }
    }

    // Create a new dimension for the result view
    let mut new_shape = D::zeros(array.ndim());
    for i in 0..array.ndim() {
        new_shape[i] = (array.shape()[i] + step[i] - 1) / step[i]; // Ceiling division
    }

    // For simplicity, we'll create a new array and copy the strided elements
    let mut result = Array::default(new_shape);

    // This is a simplified implementation that only handles 2D arrays
    // A more complete implementation would handle arbitrary dimensions
    if array.ndim() == 2 {
        let (rows, cols) = (array.shape()[0], array.shape()[1]);
        let (step_row, step_col) = (step[0], step[1]);

        let mut r = 0;
        for i in (0..rows).step_by(step_row) {
            let mut c = 0;
            for j in (0..cols).step_by(step_col) {
                result[[r, c]] = array[[i, j]].clone();
                c += 1;
            }
            r += 1;
        }
    } else {
        return Err("This simplified implementation only supports 2D arrays");
    }

    Ok(result)
}

/// Create a view of an array with a specific memory layout
///
/// # Arguments
///
/// * `array` - The input array
/// * `order` - The desired memory layout (C or F order)
///
/// # Returns
///
/// A new array with the specified memory layout
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::{as_layout, Order};
///
/// let a = array![[1, 2], [3, 4]];
/// let b = as_layout(a.view(), Order::F);
/// assert_eq!(b.shape(), &[2, 2]);
/// ```
pub fn as_layout<D, T>(array: ArrayView<T, D>, order: Order) -> Array<T, D>
where
    D: Dimension,
    T: Clone + Default,
{
    match order {
        Order::C => array.to_owned(), // ndarray is C order by default
        Order::F => {
            // Create a new array with F order
            // This is a simplified implementation that makes a copy
            let mut result = Array::zeros(array.raw_dim().f());
            result.assign(&array);
            result
        }
    }
}

/// Create a broadcast view of an array to a new shape
///
/// # Arguments
///
/// * `array` - The input array
/// * `shape` - The target shape to broadcast to
///
/// # Returns
///
/// A view of the array broadcast to the new shape, if possible
///
/// # Examples
///
/// ```
/// use ndarray::{array, Ix2};
/// use scirs2_core::ndarray_ext::broadcast_to;
///
/// let a = array![1, 2, 3];
/// let b = broadcast_to(a.view(), (3, 3)).unwrap();
/// assert_eq!(b.shape(), &[3, 3]);
/// assert_eq!(b[[0, 0]], 1);
/// assert_eq!(b[[0, 1]], 2);
/// assert_eq!(b[[1, 0]], 1);
/// ```
pub fn broadcast_to<D1, D2, T>(
    array: ArrayView<T, D1>,
    shape: D2,
) -> Result<Array<T, D2::Dim>, &'static str>
where
    D1: Dimension,
    D2: ndarray::ShapeBuilder,
    T: Clone + Default,
{
    // Convert the shape to a dimension
    let dim = shape.into_shape();
    let target_shape = dim.as_array_view();
    let src_shape = array.shape();

    // Check broadcasting rules: for each dimension, the sizes must either:
    // 1. Be the same, or
    // 2. One of them (usually the source) must be 1

    if target_shape.len() < src_shape.len() {
        return Err("Target shape cannot have fewer dimensions than source");
    }

    // Check broadcasting compatibility
    let offset = target_shape.len() - src_shape.len();
    for (i, &s) in src_shape.iter().enumerate() {
        let target_size = target_shape[i + offset];
        if s != 1 && s != target_size {
            return Err("Incompatible shapes for broadcasting");
        }
    }

    // Create the output array
    let mut result = Array::<T, _>::default(dim);

    // This is a very simplified implementation that assumes 1D to 2D broadcasting
    // A real implementation would handle arbitrary dimensions

    if src_shape.len() == 1 && target_shape.len() == 2 {
        // Broadcast 1D to 2D
        for i in 0..target_shape[0] {
            for j in 0..target_shape[1] {
                // Use j % src_shape[0] to repeat the source array elements
                let src_idx = j % src_shape[0];
                result[[i, j]] = array[src_idx].clone();
            }
        }
    } else {
        return Err("This simplified implementation only supports 1D to 2D broadcasting");
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Ix2};

    #[test]
    fn test_strided_view() {
        let a = array![[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]];
        let view = strided_view(a.view(), &[2, 2]).unwrap();
        assert_eq!(view.shape(), &[2, 2]);
        assert_eq!(view[[0, 0]], 1);
        assert_eq!(view[[0, 1]], 3);
        assert_eq!(view[[1, 0]], 9);
        assert_eq!(view[[1, 1]], 11);

        // Test invalid stride
        let result = strided_view(a.view(), &[0, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_as_layout() {
        let a = array![[1, 2], [3, 4]];

        let b = as_layout(a.view(), Order::C);
        assert_eq!(b.shape(), &[2, 2]);
        assert_eq!(b[[0, 0]], 1);
        assert_eq!(b[[1, 1]], 4);

        let c = as_layout(a.view(), Order::F);
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c[[0, 0]], 1);
        assert_eq!(c[[1, 1]], 4);
    }

    #[test]
    fn test_broadcast_to() {
        let a = array![1, 2, 3];
        let shape: Ix2 = ndarray::Ix2::from((3, 3));
        let b = broadcast_to(a.view(), shape).unwrap();
        assert_eq!(b.shape(), &[3, 3]);
        assert_eq!(b[[0, 0]], 1);
        assert_eq!(b[[0, 1]], 2);
        assert_eq!(b[[0, 2]], 3);
        assert_eq!(b[[1, 0]], 1);
        assert_eq!(b[[1, 1]], 2);
        assert_eq!(b[[1, 2]], 3);
    }
}