//! Array manipulation operations similar to `NumPy`'s array manipulation routines
//!
//! This module provides functions for manipulating arrays, including flip, roll,
//! tile, repeat, and other operations, designed to mirror `NumPy`'s functionality.

use ndarray::{Array, ArrayView, Ix1, Ix2};
use num_traits::Zero;

/// Result type for gradient function
pub type GradientResult<T> = Result<(Array<T, Ix2>, Array<T, Ix2>), &'static str>;

/// Flip a 2D array along one or more axes
///
/// # Arguments
///
/// * `array` - The input 2D array
/// * `flip_axis_0` - Whether to flip along axis 0 (rows)
/// * `flip_axis_1` - Whether to flip along axis 1 (columns)
///
/// # Returns
///
/// A new array with axes flipped as specified
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::manipulation::flip_2d;
///
/// let a = array![[1, 2, 3], [4, 5, 6]];
///
/// // Flip along rows
/// let flipped_rows = flip_2d(a.view(), true, false);
/// assert_eq!(flipped_rows, array![[4, 5, 6], [1, 2, 3]]);
///
/// // Flip along columns
/// let flipped_cols = flip_2d(a.view(), false, true);
/// assert_eq!(flipped_cols, array![[3, 2, 1], [6, 5, 4]]);
///
/// // Flip along both axes
/// let flipped_both = flip_2d(a.view(), true, true);
/// assert_eq!(flipped_both, array![[6, 5, 4], [3, 2, 1]]);
/// ```
pub fn flip_2d<T>(array: ArrayView<T, Ix2>, flip_axis_0: bool, flip_axis_1: bool) -> Array<T, Ix2>
where
    T: Clone + Zero,
{
    let (rows, cols) = (array.shape()[0], array.shape()[1]);
    let mut result = Array::<T, Ix2>::zeros((rows, cols));

    for i in 0..rows {
        for j in 0..cols {
            let src_i = if flip_axis_0 { rows - 1 - i } else { i };
            let src_j = if flip_axis_1 { cols - 1 - j } else { j };

            result[[i, j]] = array[[src_i, src_j]].clone();
        }
    }

    result
}

/// Roll array elements along one or both axes
///
/// # Arguments
///
/// * `array` - The input 2D array
/// * `shift_axis_0` - Number of places to shift along axis 0 (can be negative)
/// * `shift_axis_1` - Number of places to shift along axis 1 (can be negative)
///
/// # Returns
///
/// A new array with elements rolled as specified
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::manipulation::roll_2d;
///
/// let a = array![[1, 2, 3], [4, 5, 6]];
///
/// // Roll along rows by 1
/// let rolled_rows = roll_2d(a.view(), 1, 0);
/// assert_eq!(rolled_rows, array![[4, 5, 6], [1, 2, 3]]);
///
/// // Roll along columns by -1
/// let rolled_cols = roll_2d(a.view(), 0, -1);
/// assert_eq!(rolled_cols, array![[2, 3, 1], [5, 6, 4]]);
/// ```
pub fn roll_2d<T>(
    array: ArrayView<T, Ix2>,
    shift_axis_0: isize,
    shift_axis_1: isize,
) -> Array<T, Ix2>
where
    T: Clone + Zero,
{
    let (rows, cols) = (array.shape()[0], array.shape()[1]);

    // Handle case where no shifting is needed
    if shift_axis_0 == 0 && shift_axis_1 == 0 {
        return array.to_owned();
    }

    // Calculate effective shifts (handle negative shifts and wrap around)
    let effective_shift_0 = if rows == 0 {
        0
    } else {
        ((shift_axis_0 % rows as isize) + rows as isize) % rows as isize
    };
    let effective_shift_1 = if cols == 0 {
        0
    } else {
        ((shift_axis_1 % cols as isize) + cols as isize) % cols as isize
    };

    let mut result = Array::<T, Ix2>::zeros((rows, cols));

    for i in 0..rows {
        for j in 0..cols {
            // Calculate source indices with wrapping
            let src_i = (i as isize + rows as isize - effective_shift_0) % rows as isize;
            let src_j = (j as isize + cols as isize - effective_shift_1) % cols as isize;

            result[[i, j]] = array[[src_i as usize, src_j as usize]].clone();
        }
    }

    result
}

/// Repeat an array by tiling it in multiple dimensions
///
/// # Arguments
///
/// * `array` - The input 2D array
/// * `reps_axis_0` - Number of times to repeat the array along axis 0
/// * `reps_axis_1` - Number of times to repeat the array along axis 1
///
/// # Returns
///
/// A new array formed by repeating the input array
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::manipulation::tile_2d;
///
/// let a = array![[1, 2], [3, 4]];
///
/// // Tile array to repeat it 2 times along axis 0 and 3 times along axis 1
/// let tiled = tile_2d(a.view(), 2, 3);
/// assert_eq!(tiled.shape(), &[4, 6]);
/// assert_eq!(tiled,
///     array![
///         [1, 2, 1, 2, 1, 2],
///         [3, 4, 3, 4, 3, 4],
///         [1, 2, 1, 2, 1, 2],
///         [3, 4, 3, 4, 3, 4]
///     ]
/// );
/// ```
pub fn tile_2d<T>(array: ArrayView<T, Ix2>, reps_axis_0: usize, reps_axis_1: usize) -> Array<T, Ix2>
where
    T: Clone + Default + Zero,
{
    let (rows, cols) = (array.shape()[0], array.shape()[1]);

    // New dimensions after tiling
    let new_rows = rows * reps_axis_0;
    let new_cols = cols * reps_axis_1;

    // Edge case - zero repetitions
    if reps_axis_0 == 0 || reps_axis_1 == 0 {
        return Array::<T, Ix2>::default((0, 0));
    }

    // Edge case - one repetition
    if reps_axis_0 == 1 && reps_axis_1 == 1 {
        return array.to_owned();
    }

    let mut result = Array::<T, Ix2>::zeros((new_rows, new_cols));

    // Fill the result with repeated copies of the array
    for i in 0..new_rows {
        for j in 0..new_cols {
            let src_i = i % rows;
            let src_j = j % cols;

            result[[i, j]] = array[[src_i, src_j]].clone();
        }
    }

    result
}

/// Repeat array elements by duplicating values
///
/// # Arguments
///
/// * `array` - The input 2D array
/// * `repeats_axis_0` - Number of times to repeat each element along axis 0
/// * `repeats_axis_1` - Number of times to repeat each element along axis 1
///
/// # Returns
///
/// A new array with elements repeated as specified
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::manipulation::repeat_2d;
///
/// let a = array![[1, 2], [3, 4]];
///
/// // Repeat array elements 2 times along axis 0 and 3 times along axis 1
/// let repeated = repeat_2d(a.view(), 2, 3);
/// assert_eq!(repeated.shape(), &[4, 6]);
/// assert_eq!(repeated,
///     array![
///         [1, 1, 1, 2, 2, 2],
///         [1, 1, 1, 2, 2, 2],
///         [3, 3, 3, 4, 4, 4],
///         [3, 3, 3, 4, 4, 4]
///     ]
/// );
/// ```
pub fn repeat_2d<T>(
    array: ArrayView<T, Ix2>,
    repeats_axis_0: usize,
    repeats_axis_1: usize,
) -> Array<T, Ix2>
where
    T: Clone + Default + Zero,
{
    let (rows, cols) = (array.shape()[0], array.shape()[1]);

    // New dimensions after repeating
    let new_rows = rows * repeats_axis_0;
    let new_cols = cols * repeats_axis_1;

    // Edge case - zero repetitions
    if repeats_axis_0 == 0 || repeats_axis_1 == 0 {
        return Array::<T, Ix2>::default((0, 0));
    }

    // Edge case - one repetition
    if repeats_axis_0 == 1 && repeats_axis_1 == 1 {
        return array.to_owned();
    }

    let mut result = Array::<T, Ix2>::zeros((new_rows, new_cols));

    // Fill the result with repeated elements
    for i in 0..rows {
        for j in 0..cols {
            for i_rep in 0..repeats_axis_0 {
                for j_rep in 0..repeats_axis_1 {
                    let dest_i = i * repeats_axis_0 + i_rep;
                    let dest_j = j * repeats_axis_1 + j_rep;

                    result[[dest_i, dest_j]] = array[[i, j]].clone();
                }
            }
        }
    }

    result
}

/// Swap rows or columns in a 2D array
///
/// # Arguments
///
/// * `array` - The input 2D array
/// * `index1` - First index to swap
/// * `index2` - Second index to swap
/// * `axis` - Axis along which to swap (0 for rows, 1 for columns)
///
/// # Returns
///
/// A new array with specified rows or columns swapped
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::manipulation::swap_axes_2d;
///
/// let a = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
///
/// // Swap rows 0 and 2
/// let swapped_rows = swap_axes_2d(a.view(), 0, 2, 0).unwrap();
/// assert_eq!(swapped_rows, array![[7, 8, 9], [4, 5, 6], [1, 2, 3]]);
///
/// // Swap columns 0 and 1
/// let swapped_cols = swap_axes_2d(a.view(), 0, 1, 1).unwrap();
/// assert_eq!(swapped_cols, array![[2, 1, 3], [5, 4, 6], [8, 7, 9]]);
/// ```
pub fn swap_axes_2d<T>(
    array: ArrayView<T, Ix2>,
    index1: usize,
    index2: usize,
    axis: usize,
) -> Result<Array<T, Ix2>, &'static str>
where
    T: Clone,
{
    let (rows, cols) = (array.shape()[0], array.shape()[1]);

    if axis > 1 {
        return Err("Axis must be 0 or 1 for 2D arrays");
    }

    // Check indices are in bounds
    let axis_len = if axis == 0 { rows } else { cols };
    if index1 >= axis_len || index2 >= axis_len {
        return Err("Indices out of bounds");
    }

    // If indices are the same, just clone the array
    if index1 == index2 {
        return Ok(array.to_owned());
    }

    let mut result = array.to_owned();

    match axis {
        0 => {
            // Swap rows
            for j in 0..cols {
                let temp = result[[index1, j]].clone();
                result[[index1, j]] = result[[index2, j]].clone();
                result[[index2, j]] = temp;
            }
        }
        1 => {
            // Swap columns
            for i in 0..rows {
                let temp = result[[i, index1]].clone();
                result[[i, index1]] = result[[i, index2]].clone();
                result[[i, index2]] = temp;
            }
        }
        _ => unreachable!(),
    }

    Ok(result)
}

/// Pad a 2D array with a constant value
///
/// # Arguments
///
/// * `array` - The input 2D array
/// * `pad_width` - A tuple of tuples specifying the number of values padded
///   to the edges of each axis: ((before_axis_0, after_axis_0), (before_axis_1, after_axis_1))
/// * `pad_value` - The value to set the padded elements
///
/// # Returns
///
/// A new array with padded borders
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::manipulation::pad_2d;
///
/// let a = array![[1, 2], [3, 4]];
///
/// // Pad with 1 row before, 2 rows after, 1 column before, and 0 columns after
/// let padded = pad_2d(a.view(), ((1, 2), (1, 0)), 0);
/// assert_eq!(padded.shape(), &[5, 3]);
/// assert_eq!(padded,
///     array![
///         [0, 0, 0],
///         [0, 1, 2],
///         [0, 3, 4],
///         [0, 0, 0],
///         [0, 0, 0]
///     ]
/// );
/// ```
pub fn pad_2d<T>(
    array: ArrayView<T, Ix2>,
    pad_width: ((usize, usize), (usize, usize)),
    pad_value: T,
) -> Array<T, Ix2>
where
    T: Clone,
{
    let (rows, cols) = (array.shape()[0], array.shape()[1]);
    let ((before_0, after_0), (before_1, after_1)) = pad_width;

    // Calculate new dimensions
    let new_rows = rows + before_0 + after_0;
    let new_cols = cols + before_1 + after_1;

    // Create the result array filled with the padding value
    let mut result = Array::<T, Ix2>::from_elem((new_rows, new_cols), pad_value);

    // Copy the original array into the padded array
    for i in 0..rows {
        for j in 0..cols {
            result[[i + before_0, j + before_1]] = array[[i, j]].clone();
        }
    }

    result
}

/// Concatenate 2D arrays along a specified axis
///
/// # Arguments
///
/// * `arrays` - A slice of 2D arrays to concatenate
/// * `axis` - The axis along which to concatenate (0 for rows, 1 for columns)
///
/// # Returns
///
/// A new array containing the concatenated arrays
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::manipulation::concatenate_2d;
///
/// let a = array![[1, 2], [3, 4]];
/// let b = array![[5, 6], [7, 8]];
///
/// // Concatenate along rows (vertically)
/// let vertical = concatenate_2d(&[a.view(), b.view()], 0).unwrap();
/// assert_eq!(vertical.shape(), &[4, 2]);
/// assert_eq!(vertical, array![[1, 2], [3, 4], [5, 6], [7, 8]]);
///
/// // Concatenate along columns (horizontally)
/// let horizontal = concatenate_2d(&[a.view(), b.view()], 1).unwrap();
/// assert_eq!(horizontal.shape(), &[2, 4]);
/// assert_eq!(horizontal, array![[1, 2, 5, 6], [3, 4, 7, 8]]);
/// ```
pub fn concatenate_2d<T>(
    arrays: &[ArrayView<T, Ix2>],
    axis: usize,
) -> Result<Array<T, Ix2>, &'static str>
where
    T: Clone + Zero,
{
    if arrays.is_empty() {
        return Err("No arrays provided for concatenation");
    }

    if axis > 1 {
        return Err("Axis must be 0 or 1 for 2D arrays");
    }

    // Get the shape of the first array as a reference
    let first_shape = arrays[0].shape();

    // Calculate the total shape after concatenation
    let mut total_shape = [first_shape[0], first_shape[1]];
    for array in arrays.iter().skip(1) {
        let current_shape = array.shape();

        // Ensure all arrays have compatible shapes
        if axis == 0 && current_shape[1] != first_shape[1] {
            return Err("All arrays must have the same number of columns for axis=0 concatenation");
        } else if axis == 1 && current_shape[0] != first_shape[0] {
            return Err("All arrays must have the same number of rows for axis=1 concatenation");
        }

        total_shape[axis] += current_shape[axis];
    }

    // Create the result array
    let mut result = Array::<T, Ix2>::zeros((total_shape[0], total_shape[1]));

    // Fill the result array with data from the input arrays
    match axis {
        0 => {
            // Concatenate along axis 0 (vertically)
            let mut row_offset = 0;
            for array in arrays {
                let rows = array.shape()[0];
                let cols = array.shape()[1];

                for i in 0..rows {
                    for j in 0..cols {
                        result[[row_offset + i, j]] = array[[i, j]].clone();
                    }
                }

                row_offset += rows;
            }
        }
        1 => {
            // Concatenate along axis 1 (horizontally)
            let mut col_offset = 0;
            for array in arrays {
                let rows = array.shape()[0];
                let cols = array.shape()[1];

                for i in 0..rows {
                    for j in 0..cols {
                        result[[i, col_offset + j]] = array[[i, j]].clone();
                    }
                }

                col_offset += cols;
            }
        }
        _ => unreachable!(),
    }

    Ok(result)
}

/// Stack a sequence of 1D arrays into a 2D array
///
/// # Arguments
///
/// * `arrays` - A slice of 1D arrays to stack
///
/// # Returns
///
/// A 2D array where each row contains an input array
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::manipulation::vstack_1d;
///
/// let a = array![1, 2, 3];
/// let b = array![4, 5, 6];
/// let c = array![7, 8, 9];
///
/// let stacked = vstack_1d(&[a.view(), b.view(), c.view()]).unwrap();
/// assert_eq!(stacked.shape(), &[3, 3]);
/// assert_eq!(stacked, array![[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
/// ```
pub fn vstack_1d<T>(arrays: &[ArrayView<T, Ix1>]) -> Result<Array<T, Ix2>, &'static str>
where
    T: Clone + Zero,
{
    if arrays.is_empty() {
        return Err("No arrays provided for stacking");
    }

    // All arrays must have the same length
    let expected_len = arrays[0].len();
    for (_i, array) in arrays.iter().enumerate().skip(1) {
        if array.len() != expected_len {
            return Err("Arrays must have consistent lengths for stacking");
        }
    }

    // Create the result array
    let rows = arrays.len();
    let cols = expected_len;
    let mut result = Array::<T, Ix2>::zeros((rows, cols));

    // Fill the result array
    for (i, array) in arrays.iter().enumerate() {
        for (j, val) in array.iter().enumerate() {
            result[[i, j]] = val.clone();
        }
    }

    Ok(result)
}

/// Stack a sequence of 1D arrays horizontally (as columns) into a 2D array
///
/// # Arguments
///
/// * `arrays` - A slice of 1D arrays to stack
///
/// # Returns
///
/// A 2D array where each column contains an input array
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::manipulation::hstack_1d;
///
/// let a = array![1, 2, 3];
/// let b = array![4, 5, 6];
///
/// let stacked = hstack_1d(&[a.view(), b.view()]).unwrap();
/// assert_eq!(stacked.shape(), &[3, 2]);
/// assert_eq!(stacked, array![[1, 4], [2, 5], [3, 6]]);
/// ```
pub fn hstack_1d<T>(arrays: &[ArrayView<T, Ix1>]) -> Result<Array<T, Ix2>, &'static str>
where
    T: Clone + Zero,
{
    if arrays.is_empty() {
        return Err("No arrays provided for stacking");
    }

    // All arrays must have the same length
    let expected_len = arrays[0].len();
    for (_i, array) in arrays.iter().enumerate().skip(1) {
        if array.len() != expected_len {
            return Err("Arrays must have consistent lengths for stacking");
        }
    }

    // Create the result array
    let rows = expected_len;
    let cols = arrays.len();
    let mut result = Array::<T, Ix2>::zeros((rows, cols));

    // Fill the result array
    for (j, array) in arrays.iter().enumerate() {
        for (i, val) in array.iter().enumerate() {
            result[[i, j]] = val.clone();
        }
    }

    Ok(result)
}

/// Remove a dimension of size 1 from a 2D array, resulting in a 1D array
///
/// # Arguments
///
/// * `array` - The input 2D array
/// * `axis` - The axis to squeeze (0 for rows, 1 for columns)
///
/// # Returns
///
/// A 1D array with the specified dimension removed
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::manipulation::squeeze_2d;
///
/// let a = array![[1, 2, 3]];  // 1x3 array (1 row, 3 columns)
/// let b = array![[1], [2], [3]];  // 3x1 array (3 rows, 1 column)
///
/// // Squeeze out the row dimension (axis 0) from a
/// let squeezed_a = squeeze_2d(a.view(), 0).unwrap();
/// assert_eq!(squeezed_a.shape(), &[3]);
/// assert_eq!(squeezed_a, array![1, 2, 3]);
///
/// // Squeeze out the column dimension (axis 1) from b
/// let squeezed_b = squeeze_2d(b.view(), 1).unwrap();
/// assert_eq!(squeezed_b.shape(), &[3]);
/// assert_eq!(squeezed_b, array![1, 2, 3]);
/// ```
pub fn squeeze_2d<T>(array: ArrayView<T, Ix2>, axis: usize) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Zero,
{
    let (rows, cols) = (array.shape()[0], array.shape()[1]);

    match axis {
        0 => {
            // Squeeze out row dimension
            if rows != 1 {
                return Err("Cannot squeeze array with more than 1 row along axis 0");
            }

            let mut result = Array::<T, Ix1>::zeros(cols);
            for j in 0..cols {
                result[j] = array[[0, j]].clone();
            }

            Ok(result)
        }
        1 => {
            // Squeeze out column dimension
            if cols != 1 {
                return Err("Cannot squeeze array with more than 1 column along axis 1");
            }

            let mut result = Array::<T, Ix1>::zeros(rows);
            for i in 0..rows {
                result[i] = array[[i, 0]].clone();
            }

            Ok(result)
        }
        _ => Err("Axis must be 0 or 1 for 2D arrays"),
    }
}

/// Create a meshgrid from 1D coordinate arrays
///
/// # Arguments
///
/// * `x` - 1D array of x coordinates
/// * `y` - 1D array of y coordinates
///
/// # Returns
///
/// A tuple of two 2D arrays (X, Y) where X and Y are copies of the input arrays
/// repeated to form a meshgrid
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::manipulation::meshgrid;
///
/// let x = array![1, 2, 3];
/// let y = array![4, 5];
/// let (x_grid, y_grid) = meshgrid(x.view(), y.view()).unwrap();
/// assert_eq!(x_grid.shape(), &[2, 3]);
/// assert_eq!(y_grid.shape(), &[2, 3]);
/// assert_eq!(x_grid, array![[1, 2, 3], [1, 2, 3]]);
/// assert_eq!(y_grid, array![[4, 4, 4], [5, 5, 5]]);
/// ```
pub fn meshgrid<T>(x: ArrayView<T, Ix1>, y: ArrayView<T, Ix1>) -> GradientResult<T>
where
    T: Clone + Zero,
{
    let nx = x.len();
    let ny = y.len();

    if nx == 0 || ny == 0 {
        return Err("Input arrays must not be empty");
    }

    // Create output arrays
    let mut x_grid = Array::<T, Ix2>::zeros((ny, nx));
    let mut y_grid = Array::<T, Ix2>::zeros((ny, nx));

    // Fill the meshgrid
    for i in 0..ny {
        for j in 0..nx {
            x_grid[[i, j]] = x[j].clone();
            y_grid[[i, j]] = y[i].clone();
        }
    }

    Ok((x_grid, y_grid))
}

/// Find unique elements in an array
///
/// # Arguments
///
/// * `array` - The input 1D array
///
/// # Returns
///
/// A 1D array containing the unique elements of the input array, sorted
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::manipulation::unique;
///
/// let a = array![3, 1, 2, 2, 3, 4, 1];
/// let result = unique(a.view()).unwrap();
/// assert_eq!(result, array![1, 2, 3, 4]);
/// ```
pub fn unique<T>(array: ArrayView<T, Ix1>) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Ord,
{
    if array.is_empty() {
        return Err("Input array must not be empty");
    }

    // Clone elements to a Vec and sort
    let mut values: Vec<T> = array.iter().cloned().collect();
    values.sort();

    // Remove duplicates
    values.dedup();

    // Convert to ndarray
    Ok(Array::from_vec(values))
}

/// Return the indices of the minimum values along the specified axis
///
/// # Arguments
///
/// * `array` - The input 2D array
/// * `axis` - The axis along which to find the minimum values (0 for rows, 1 for columns, None for flattened array)
///
/// # Returns
///
/// A 1D array containing the indices of the minimum values
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::manipulation::argmin;
///
/// let a = array![[5, 2, 3], [4, 1, 6]];
///
/// // Find indices of minimum values along axis 0 (columns)
/// let result = argmin(a.view(), Some(0)).unwrap();
/// assert_eq!(result, array![1, 1, 0]); // The indices of min values in each column
///
/// // Find indices of minimum values along axis 1 (rows)
/// let result = argmin(a.view(), Some(1)).unwrap();
/// assert_eq!(result, array![1, 1]); // The indices of min values in each row
///
/// // Find index of minimum value in flattened array
/// let result = argmin(a.view(), None).unwrap();
/// assert_eq!(result[0], 4); // The index of the minimum value in the flattened array (row 1, col 1)
/// ```
pub fn argmin<T>(
    array: ArrayView<T, Ix2>,
    axis: Option<usize>,
) -> Result<Array<usize, Ix1>, &'static str>
where
    T: Clone + PartialOrd,
{
    let (rows, cols) = (array.shape()[0], array.shape()[1]);

    if rows == 0 || cols == 0 {
        return Err("Input array must not be empty");
    }

    match axis {
        Some(0) => {
            // Find min indices along axis 0 (for each column)
            let mut indices = Array::<usize, Ix1>::zeros(cols);

            for j in 0..cols {
                let mut min_idx = 0;
                let mut min_val = &array[[0, j]];

                for i in 1..rows {
                    if &array[[i, j]] < min_val {
                        min_idx = i;
                        min_val = &array[[i, j]];
                    }
                }

                indices[j] = min_idx;
            }

            Ok(indices)
        }
        Some(1) => {
            // Find min indices along axis 1 (for each row)
            let mut indices = Array::<usize, Ix1>::zeros(rows);

            for i in 0..rows {
                let mut min_idx = 0;
                let mut min_val = &array[[i, 0]];

                for j in 1..cols {
                    if &array[[i, j]] < min_val {
                        min_idx = j;
                        min_val = &array[[i, j]];
                    }
                }

                indices[i] = min_idx;
            }

            Ok(indices)
        }
        Some(_) => Err("Axis must be 0 or 1 for 2D arrays"),
        None => {
            // Find min index in flattened array
            let mut min_idx = 0;
            let mut min_val = &array[[0, 0]];

            for i in 0..rows {
                for j in 0..cols {
                    if &array[[i, j]] < min_val {
                        min_idx = i * cols + j;
                        min_val = &array[[i, j]];
                    }
                }
            }

            Ok(Array::from_vec(vec![min_idx]))
        }
    }
}

/// Return the indices of the maximum values along the specified axis
///
/// # Arguments
///
/// * `array` - The input 2D array
/// * `axis` - The axis along which to find the maximum values (0 for rows, 1 for columns, None for flattened array)
///
/// # Returns
///
/// A 1D array containing the indices of the maximum values
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::manipulation::argmax;
///
/// let a = array![[5, 2, 3], [4, 1, 6]];
///
/// // Find indices of maximum values along axis 0 (columns)
/// let result = argmax(a.view(), Some(0)).unwrap();
/// assert_eq!(result, array![0, 0, 1]); // The indices of max values in each column
///
/// // Find indices of maximum values along axis 1 (rows)
/// let result = argmax(a.view(), Some(1)).unwrap();
/// assert_eq!(result, array![0, 2]); // The indices of max values in each row
///
/// // Find index of maximum value in flattened array
/// let result = argmax(a.view(), None).unwrap();
/// assert_eq!(result[0], 5); // The index of the maximum value in the flattened array (row 1, col 2)
/// ```
pub fn argmax<T>(
    array: ArrayView<T, Ix2>,
    axis: Option<usize>,
) -> Result<Array<usize, Ix1>, &'static str>
where
    T: Clone + PartialOrd,
{
    let (rows, cols) = (array.shape()[0], array.shape()[1]);

    if rows == 0 || cols == 0 {
        return Err("Input array must not be empty");
    }

    match axis {
        Some(0) => {
            // Find max indices along axis 0 (for each column)
            let mut indices = Array::<usize, Ix1>::zeros(cols);

            for j in 0..cols {
                let mut max_idx = 0;
                let mut max_val = &array[[0, j]];

                for i in 1..rows {
                    if &array[[i, j]] > max_val {
                        max_idx = i;
                        max_val = &array[[i, j]];
                    }
                }

                indices[j] = max_idx;
            }

            Ok(indices)
        }
        Some(1) => {
            // Find max indices along axis 1 (for each row)
            let mut indices = Array::<usize, Ix1>::zeros(rows);

            for i in 0..rows {
                let mut max_idx = 0;
                let mut max_val = &array[[i, 0]];

                for j in 1..cols {
                    if &array[[i, j]] > max_val {
                        max_idx = j;
                        max_val = &array[[i, j]];
                    }
                }

                indices[i] = max_idx;
            }

            Ok(indices)
        }
        Some(_) => Err("Axis must be 0 or 1 for 2D arrays"),
        None => {
            // Find max index in flattened array
            let mut max_idx = 0;
            let mut max_val = &array[[0, 0]];

            for i in 0..rows {
                for j in 0..cols {
                    if &array[[i, j]] > max_val {
                        max_idx = i * cols + j;
                        max_val = &array[[i, j]];
                    }
                }
            }

            Ok(Array::from_vec(vec![max_idx]))
        }
    }
}

/// Calculate the gradient of an array
///
/// # Arguments
///
/// * `array` - The input 2D array
/// * `spacing` - Optional tuple of spacings for each axis
///
/// # Returns
///
/// A tuple of arrays (grad_y, grad_x) containing the gradient along each axis
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::manipulation::gradient;
///
/// let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
///
/// // Calculate gradient with default spacing
/// let (grad_y, grad_x) = gradient(a.view(), None).unwrap();
/// // Vertical gradient (y-direction)
/// assert_eq!(grad_y.shape(), &[2, 3]);
/// // Horizontal gradient (x-direction)
/// assert_eq!(grad_x.shape(), &[2, 3]);
/// ```
pub fn gradient<T>(array: ArrayView<T, Ix2>, spacing: Option<(T, T)>) -> GradientResult<T>
where
    T: Clone + num_traits::Float,
{
    let (rows, cols) = (array.shape()[0], array.shape()[1]);

    if rows == 0 || cols == 0 {
        return Err("Input array must not be empty");
    }

    // Get spacing values (default to 1.0)
    let (dy, dx) = spacing.unwrap_or((T::one(), T::one()));

    // Create output arrays for gradients
    let mut grad_y = Array::<T, Ix2>::zeros((rows, cols));
    let mut grad_x = Array::<T, Ix2>::zeros((rows, cols));

    // Calculate gradient along y axis (rows)
    if rows == 1 {
        // Single row, gradient is zero
        // (already initialized with zeros)
    } else {
        // First row: forward difference
        for j in 0..cols {
            grad_y[[0, j]] = (array[[1, j]] - array[[0, j]]) / dy;
        }

        // Middle rows: central difference
        for i in 1..rows - 1 {
            for j in 0..cols {
                grad_y[[i, j]] = (array[[i + 1, j]] - array[[i - 1, j]]) / (dy + dy);
            }
        }

        // Last row: backward difference
        for j in 0..cols {
            grad_y[[rows - 1, j]] = (array[[rows - 1, j]] - array[[rows - 2, j]]) / dy;
        }
    }

    // Calculate gradient along x axis (columns)
    if cols == 1 {
        // Single column, gradient is zero
        // (already initialized with zeros)
    } else {
        for i in 0..rows {
            // First column: forward difference
            grad_x[[i, 0]] = (array[[i, 1]] - array[[i, 0]]) / dx;

            // Middle columns: central difference
            for j in 1..cols - 1 {
                grad_x[[i, j]] = (array[[i, j + 1]] - array[[i, j - 1]]) / (dx + dx);
            }

            // Last column: backward difference
            grad_x[[i, cols - 1]] = (array[[i, cols - 1]] - array[[i, cols - 2]]) / dx;
        }
    }

    Ok((grad_y, grad_x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_flip_2d() {
        let a = array![[1, 2, 3], [4, 5, 6]];

        // Test flipping along axis 0 (rows)
        let flipped_rows = flip_2d(a.view(), true, false);
        assert_eq!(flipped_rows, array![[4, 5, 6], [1, 2, 3]]);

        // Test flipping along axis 1 (columns)
        let flipped_cols = flip_2d(a.view(), false, true);
        assert_eq!(flipped_cols, array![[3, 2, 1], [6, 5, 4]]);

        // Test flipping along both axes
        let flipped_both = flip_2d(a.view(), true, true);
        assert_eq!(flipped_both, array![[6, 5, 4], [3, 2, 1]]);
    }

    #[test]
    fn test_roll_2d() {
        let a = array![[1, 2, 3], [4, 5, 6]];

        // Test rolling along axis 0 (rows)
        let rolled_rows = roll_2d(a.view(), 1, 0);
        assert_eq!(rolled_rows, array![[4, 5, 6], [1, 2, 3]]);

        // Test rolling along axis 1 (columns)
        let rolled_cols = roll_2d(a.view(), 0, 1);
        assert_eq!(rolled_cols, array![[3, 1, 2], [6, 4, 5]]);

        // Test negative rolling
        let rolled_neg = roll_2d(a.view(), 0, -1);
        assert_eq!(rolled_neg, array![[2, 3, 1], [5, 6, 4]]);

        // Test rolling by zero (should return the original array)
        let rolled_zero = roll_2d(a.view(), 0, 0);
        assert_eq!(rolled_zero, a);
    }

    #[test]
    fn test_tile_2d() {
        let a = array![[1, 2], [3, 4]];

        // Test tiling along both axes
        let tiled = tile_2d(a.view(), 2, 3);
        assert_eq!(tiled.shape(), &[4, 6]);
        assert_eq!(
            tiled,
            array![
                [1, 2, 1, 2, 1, 2],
                [3, 4, 3, 4, 3, 4],
                [1, 2, 1, 2, 1, 2],
                [3, 4, 3, 4, 3, 4]
            ]
        );

        // Test tiling along axis 0 only
        let tiled_axis_0 = tile_2d(a.view(), 2, 1);
        assert_eq!(tiled_axis_0.shape(), &[4, 2]);
        assert_eq!(tiled_axis_0, array![[1, 2], [3, 4], [1, 2], [3, 4]]);

        // Test tiling a single element
        let single = array![[5]];
        let tiled_single = tile_2d(single.view(), 2, 2);
        assert_eq!(tiled_single.shape(), &[2, 2]);
        assert_eq!(tiled_single, array![[5, 5], [5, 5]]);
    }

    #[test]
    fn test_repeat_2d() {
        let a = array![[1, 2], [3, 4]];

        // Test repeating along both axes
        let repeated = repeat_2d(a.view(), 2, 3);
        assert_eq!(repeated.shape(), &[4, 6]);
        assert_eq!(
            repeated,
            array![
                [1, 1, 1, 2, 2, 2],
                [1, 1, 1, 2, 2, 2],
                [3, 3, 3, 4, 4, 4],
                [3, 3, 3, 4, 4, 4]
            ]
        );

        // Test repeating along axis 1 only
        let repeated_axis_1 = repeat_2d(a.view(), 1, 2);
        assert_eq!(repeated_axis_1.shape(), &[2, 4]);
        assert_eq!(repeated_axis_1, array![[1, 1, 2, 2], [3, 3, 4, 4]]);
    }

    #[test]
    fn test_swap_axes_2d() {
        let a = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];

        // Test swapping rows
        let swapped_rows = swap_axes_2d(a.view(), 0, 2, 0).unwrap();
        assert_eq!(swapped_rows, array![[7, 8, 9], [4, 5, 6], [1, 2, 3]]);

        // Test swapping columns
        let swapped_cols = swap_axes_2d(a.view(), 0, 2, 1).unwrap();
        assert_eq!(swapped_cols, array![[3, 2, 1], [6, 5, 4], [9, 8, 7]]);

        // Test swapping same indices (should return a clone of the original)
        let swapped_same = swap_axes_2d(a.view(), 1, 1, 0).unwrap();
        assert_eq!(swapped_same, a);

        // Test invalid axis
        assert!(swap_axes_2d(a.view(), 0, 1, 2).is_err());

        // Test out of bounds indices
        assert!(swap_axes_2d(a.view(), 0, 3, 0).is_err());
    }

    #[test]
    fn test_pad_2d() {
        let a = array![[1, 2], [3, 4]];

        // Test padding on all sides
        let padded_all = pad_2d(a.view(), ((1, 1), (1, 1)), 0);
        assert_eq!(padded_all.shape(), &[4, 4]);
        assert_eq!(
            padded_all,
            array![[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]]
        );

        // Test uneven padding
        let padded_uneven = pad_2d(a.view(), ((2, 0), (0, 1)), 9);
        assert_eq!(padded_uneven.shape(), &[4, 3]);
        assert_eq!(
            padded_uneven,
            array![[9, 9, 9], [9, 9, 9], [1, 2, 9], [3, 4, 9]]
        );
    }

    #[test]
    fn test_concatenate_2d() {
        let a = array![[1, 2], [3, 4]];
        let b = array![[5, 6], [7, 8]];

        // Test concatenating along axis 0 (vertically)
        let vertical = concatenate_2d(&[a.view(), b.view()], 0).unwrap();
        assert_eq!(vertical.shape(), &[4, 2]);
        assert_eq!(vertical, array![[1, 2], [3, 4], [5, 6], [7, 8]]);

        // Test concatenating along axis 1 (horizontally)
        let horizontal = concatenate_2d(&[a.view(), b.view()], 1).unwrap();
        assert_eq!(horizontal.shape(), &[2, 4]);
        assert_eq!(horizontal, array![[1, 2, 5, 6], [3, 4, 7, 8]]);

        // Test concatenating with incompatible shapes
        let c = array![[9, 10, 11]];
        assert!(concatenate_2d(&[a.view(), c.view()], 0).is_err());

        // Test concatenating empty array list
        let empty: [ArrayView<i32, Ix2>; 0] = [];
        assert!(concatenate_2d(&empty, 0).is_err());

        // Test invalid axis
        assert!(concatenate_2d(&[a.view(), b.view()], 2).is_err());
    }

    #[test]
    fn test_vstack_1d() {
        let a = array![1, 2, 3];
        let b = array![4, 5, 6];
        let c = array![7, 8, 9];

        // Test stacking multiple arrays
        let stacked = vstack_1d(&[a.view(), b.view(), c.view()]).unwrap();
        assert_eq!(stacked.shape(), &[3, 3]);
        assert_eq!(stacked, array![[1, 2, 3], [4, 5, 6], [7, 8, 9]]);

        // Test stacking empty list
        let empty: [ArrayView<i32, Ix1>; 0] = [];
        assert!(vstack_1d(&empty).is_err());

        // Test inconsistent lengths
        let d = array![10, 11];
        assert!(vstack_1d(&[a.view(), d.view()]).is_err());
    }

    #[test]
    fn test_hstack_1d() {
        let a = array![1, 2, 3];
        let b = array![4, 5, 6];

        // Test stacking multiple arrays
        let stacked = hstack_1d(&[a.view(), b.view()]).unwrap();
        assert_eq!(stacked.shape(), &[3, 2]);
        assert_eq!(stacked, array![[1, 4], [2, 5], [3, 6]]);

        // Test stacking empty list
        let empty: [ArrayView<i32, Ix1>; 0] = [];
        assert!(hstack_1d(&empty).is_err());

        // Test inconsistent lengths
        let c = array![7, 8];
        assert!(hstack_1d(&[a.view(), c.view()]).is_err());
    }

    #[test]
    fn test_squeeze_2d() {
        let a = array![[1, 2, 3]]; // 1x3 array
        let b = array![[1], [2], [3]]; // 3x1 array

        // Test squeezing axis 0
        let squeezed_a = squeeze_2d(a.view(), 0).unwrap();
        assert_eq!(squeezed_a.shape(), &[3]);
        assert_eq!(squeezed_a, array![1, 2, 3]);

        // Test squeezing axis 1
        let squeezed_b = squeeze_2d(b.view(), 1).unwrap();
        assert_eq!(squeezed_b.shape(), &[3]);
        assert_eq!(squeezed_b, array![1, 2, 3]);

        // Test squeezing on an axis with size > 1 (should fail)
        let c = array![[1, 2], [3, 4]]; // 2x2 array
        assert!(squeeze_2d(c.view(), 0).is_err());
        assert!(squeeze_2d(c.view(), 1).is_err());

        // Test invalid axis
        assert!(squeeze_2d(a.view(), 2).is_err());
    }

    #[test]
    fn test_meshgrid() {
        let x = array![1, 2, 3];
        let y = array![4, 5];

        let (x_grid, y_grid) = meshgrid(x.view(), y.view()).unwrap();
        assert_eq!(x_grid.shape(), &[2, 3]);
        assert_eq!(y_grid.shape(), &[2, 3]);
        assert_eq!(x_grid, array![[1, 2, 3], [1, 2, 3]]);
        assert_eq!(y_grid, array![[4, 4, 4], [5, 5, 5]]);

        // Test empty arrays
        let empty = array![];
        assert!(meshgrid(x.view(), empty.view()).is_err());
        assert!(meshgrid(empty.view(), y.view()).is_err());
    }

    #[test]
    fn test_unique() {
        let a = array![3, 1, 2, 2, 3, 4, 1];
        let result = unique(a.view()).unwrap();
        assert_eq!(result, array![1, 2, 3, 4]);

        // Test empty array
        let empty: Array<i32, Ix1> = array![];
        assert!(unique(empty.view()).is_err());
    }

    #[test]
    fn test_argmin() {
        let a = array![[5, 2, 3], [4, 1, 6]];

        // Test along axis 0
        let result = argmin(a.view(), Some(0)).unwrap();
        assert_eq!(result, array![1, 1, 0]);

        // Test along axis 1
        let result = argmin(a.view(), Some(1)).unwrap();
        assert_eq!(result, array![1, 1]);

        // Test flattened array
        let result = argmin(a.view(), None).unwrap();
        assert_eq!(result[0], 4); // Index of 1 in the flattened array (row 1, col 1)

        // Test invalid axis
        assert!(argmin(a.view(), Some(2)).is_err());

        // Test empty array
        let empty: Array<i32, Ix2> = Array::zeros((0, 0));
        assert!(argmin(empty.view(), Some(0)).is_err());
    }

    #[test]
    fn test_argmax() {
        let a = array![[5, 2, 3], [4, 1, 6]];

        // Test along axis 0
        let result = argmax(a.view(), Some(0)).unwrap();
        assert_eq!(result, array![0, 0, 1]);

        // Test along axis 1
        let result = argmax(a.view(), Some(1)).unwrap();
        assert_eq!(result, array![0, 2]);

        // Test flattened array
        let result = argmax(a.view(), None).unwrap();
        assert_eq!(result[0], 5); // Index of 6 in the flattened array (row 1, col 2)

        // Test invalid axis
        assert!(argmax(a.view(), Some(2)).is_err());

        // Test empty array
        let empty: Array<i32, Ix2> = Array::zeros((0, 0));
        assert!(argmax(empty.view(), Some(0)).is_err());
    }

    #[test]
    fn test_gradient() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        // Calculate gradient with default spacing
        let (grad_y, grad_x) = gradient(a.view(), None).unwrap();

        // Verify shapes
        assert_eq!(grad_y.shape(), &[2, 3]);
        assert_eq!(grad_x.shape(), &[2, 3]);

        // Check gradient values
        // Vertical gradient (y-direction)
        assert_abs_diff_eq!(grad_y[[0, 0]], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(grad_y[[0, 1]], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(grad_y[[0, 2]], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(grad_y[[1, 0]], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(grad_y[[1, 1]], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(grad_y[[1, 2]], 3.0, epsilon = 1e-10);

        // Horizontal gradient (x-direction)
        assert_abs_diff_eq!(grad_x[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(grad_x[[0, 1]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(grad_x[[0, 2]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(grad_x[[1, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(grad_x[[1, 1]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(grad_x[[1, 2]], 1.0, epsilon = 1e-10);

        // Test with custom spacing
        let (grad_y, grad_x) = gradient(a.view(), Some((2.0, 0.5))).unwrap();

        // Vertical gradient (y-direction) with spacing = 2.0
        assert_abs_diff_eq!(grad_y[[0, 0]], 1.5, epsilon = 1e-10); // 3.0 / 2.0
        assert_abs_diff_eq!(grad_y[[0, 1]], 1.5, epsilon = 1e-10);
        assert_abs_diff_eq!(grad_y[[0, 2]], 1.5, epsilon = 1e-10);

        // Horizontal gradient (x-direction) with spacing = 0.5
        assert_abs_diff_eq!(grad_x[[0, 0]], 2.0, epsilon = 1e-10); // 1.0 / 0.5
        assert_abs_diff_eq!(grad_x[[0, 1]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(grad_x[[0, 2]], 2.0, epsilon = 1e-10);

        // Test empty array
        let empty: Array<f32, Ix2> = Array::zeros((0, 0));
        assert!(gradient(empty.view(), None).is_err());
    }
}
