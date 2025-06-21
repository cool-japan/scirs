//! Advanced indexing operations for ndarray
//!
//! This module provides enhanced indexing capabilities including boolean
//! masking, fancy indexing, and advanced slicing operations similar to
//! `NumPy`'s advanced indexing functionality.

use ndarray::{Array, ArrayView, Ix1, Ix2};

/// Result type for coordinating indices
pub type IndicesResult = Result<(Array<usize, Ix1>, Array<usize, Ix1>), &'static str>;

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
/// A 2D array of values at the specified indices along the given axis
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::indexing::take_2d;
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

/// Boolean mask indexing for 2D arrays
///
/// # Arguments
///
/// * `array` - The input 2D array
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
/// use scirs2_core::ndarray_ext::indexing::boolean_mask_2d;
///
/// let a = array![[1, 2, 3], [4, 5, 6]];
/// let mask = array![[true, false, true], [false, true, false]];
/// let result = boolean_mask_2d(a.view(), mask.view()).unwrap();
/// assert_eq!(result.len(), 3);
/// assert_eq!(result[0], 1);
/// assert_eq!(result[1], 3);
/// assert_eq!(result[2], 5);
/// ```
pub fn boolean_mask_2d<T>(
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

/// Boolean mask indexing for 1D arrays
///
/// # Arguments
///
/// * `array` - The input 1D array
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
/// use scirs2_core::ndarray_ext::indexing::boolean_mask_1d;
///
/// let a = array![1, 2, 3, 4, 5];
/// let mask = array![true, false, true, false, true];
/// let result = boolean_mask_1d(a.view(), mask.view()).unwrap();
/// assert_eq!(result.len(), 3);
/// assert_eq!(result[0], 1);
/// assert_eq!(result[1], 3);
/// assert_eq!(result[2], 5);
/// ```
pub fn boolean_mask_1d<T>(
    array: ArrayView<T, Ix1>,
    mask: ArrayView<bool, Ix1>,
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

/// Indexed slicing for 1D arrays
///
/// # Arguments
///
/// * `array` - The input 1D array
/// * `indices` - Array of indices to extract
///
/// # Returns
///
/// A 1D array containing the elements at the specified indices
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::indexing::take_1d;
///
/// let a = array![10, 20, 30, 40, 50];
/// let indices = array![0, 2, 4];
/// let result = take_1d(a.view(), indices.view()).unwrap();
/// assert_eq!(result.len(), 3);
/// assert_eq!(result[0], 10);
/// assert_eq!(result[1], 30);
/// assert_eq!(result[2], 50);
/// ```
pub fn take_1d<T>(
    array: ArrayView<T, Ix1>,
    indices: ArrayView<usize, Ix1>,
) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Default,
{
    let len = array.len();

    // Verify that indices are in bounds
    for &idx in indices.iter() {
        if idx >= len {
            return Err("Index out of bounds");
        }
    }

    // Create the result array
    let mut result = Array::<T, Ix1>::default(indices.len());

    // Extract the elements at the specified indices
    for (i, &idx) in indices.iter().enumerate() {
        result[i] = array[idx].clone();
    }

    Ok(result)
}

/// Fancy indexing for 2D arrays with pairs of index arrays
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
/// use scirs2_core::ndarray_ext::indexing::fancy_index_2d;
///
/// let a = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
/// let row_indices = array![0, 1, 2];
/// let col_indices = array![0, 1, 2];
/// let result = fancy_index_2d(a.view(), row_indices.view(), col_indices.view()).unwrap();
/// assert_eq!(result.len(), 3);
/// assert_eq!(result[0], 1);
/// assert_eq!(result[1], 5);
/// assert_eq!(result[2], 9);
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

/// Extract a diagonal or a sub-diagonal from a 2D array
///
/// # Arguments
///
/// * `array` - The input 2D array
/// * `offset` - Offset from the main diagonal (0 for main diagonal, positive for above, negative for below)
///
/// # Returns
///
/// A 1D array containing the diagonal elements
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::indexing::diagonal;
///
/// let a = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
///
/// // Main diagonal
/// let main_diag = diagonal(a.view(), 0).unwrap();
/// assert_eq!(main_diag.len(), 3);
/// assert_eq!(main_diag[0], 1);
/// assert_eq!(main_diag[1], 5);
/// assert_eq!(main_diag[2], 9);
///
/// // Upper diagonal
/// let upper_diag = diagonal(a.view(), 1).unwrap();
/// assert_eq!(upper_diag.len(), 2);
/// assert_eq!(upper_diag[0], 2);
/// assert_eq!(upper_diag[1], 6);
///
/// // Lower diagonal
/// let lower_diag = diagonal(a.view(), -1).unwrap();
/// assert_eq!(lower_diag.len(), 2);
/// assert_eq!(lower_diag[0], 4);
/// assert_eq!(lower_diag[1], 8);
/// ```
pub fn diagonal<T>(array: ArrayView<T, Ix2>, offset: isize) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Default,
{
    let (rows, cols) = (array.shape()[0], array.shape()[1]);

    // Calculate the length of the diagonal
    let diag_len = if offset >= 0 {
        std::cmp::min(rows, cols.saturating_sub(offset as usize))
    } else {
        std::cmp::min(cols, rows.saturating_sub((-offset) as usize))
    };

    if diag_len == 0 {
        return Err("No diagonal elements for the given offset");
    }

    // Create the result array
    let mut result = Array::<T, Ix1>::default(diag_len);

    // Extract the diagonal elements
    for i in 0..diag_len {
        let row = if offset < 0 {
            i + (-offset) as usize
        } else {
            i
        };

        let col = if offset > 0 { i + offset as usize } else { i };

        result[i] = array[[row, col]].clone();
    }

    Ok(result)
}

/// Where function - select elements based on a condition for 1D arrays
///
/// # Arguments
///
/// * `array` - The input 1D array
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
/// use scirs2_core::ndarray_ext::indexing::where_1d;
///
/// let a = array![1, 2, 3, 4, 5];
/// let result = where_1d(a.view(), |&x| x > 3).unwrap();
/// assert_eq!(result.len(), 2);
/// assert_eq!(result[0], 4);
/// assert_eq!(result[1], 5);
/// ```
pub fn where_1d<T, F>(array: ArrayView<T, Ix1>, condition: F) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Default,
    F: Fn(&T) -> bool,
{
    // Build a boolean mask array based on the condition
    let mask = array.map(condition);

    // Use the boolean_mask_1d function to select elements
    boolean_mask_1d(array, mask.view())
}

/// Where function - select elements based on a condition for 2D arrays
///
/// # Arguments
///
/// * `array` - The input 2D array
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
/// use scirs2_core::ndarray_ext::indexing::where_2d;
///
/// let a = array![[1, 2, 3], [4, 5, 6]];
/// let result = where_2d(a.view(), |&x| x > 3).unwrap();
/// assert_eq!(result.len(), 3);
/// assert_eq!(result[0], 4);
/// assert_eq!(result[1], 5);
/// assert_eq!(result[2], 6);
/// ```
pub fn where_2d<T, F>(array: ArrayView<T, Ix2>, condition: F) -> Result<Array<T, Ix1>, &'static str>
where
    T: Clone + Default,
    F: Fn(&T) -> bool,
{
    // Build a boolean mask array based on the condition
    let mask = array.map(condition);

    // Use the boolean_mask_2d function to select elements
    boolean_mask_2d(array, mask.view())
}

/// Extract indices where a 1D array meets a condition
///
/// # Arguments
///
/// * `array` - The input 1D array
/// * `condition` - A function that takes a reference to an element and returns a bool
///
/// # Returns
///
/// A 1D array of indices where the condition is true
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::indexing::indices_where_1d;
///
/// let a = array![10, 20, 30, 40, 50];
/// let result = indices_where_1d(a.view(), |&x| x > 30).unwrap();
/// assert_eq!(result.len(), 2);
/// assert_eq!(result[0], 3);
/// assert_eq!(result[1], 4);
/// ```
pub fn indices_where_1d<T, F>(
    array: ArrayView<T, Ix1>,
    condition: F,
) -> Result<Array<usize, Ix1>, &'static str>
where
    T: Clone,
    F: Fn(&T) -> bool,
{
    // Build a vector of indices where the condition is true
    let mut indices = Vec::new();

    for (i, val) in array.iter().enumerate() {
        if condition(val) {
            indices.push(i);
        }
    }

    // Convert the vector to an ndarray Array
    Ok(Array::from_vec(indices))
}

/// Extract indices where a 2D array meets a condition
///
/// # Arguments
///
/// * `array` - The input 2D array
/// * `condition` - A function that takes a reference to an element and returns a bool
///
/// # Returns
///
/// A tuple of two 1D arrays (row_indices, col_indices) where the condition is true
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::indexing::indices_where_2d;
///
/// let a = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
/// let (rows, cols) = indices_where_2d(a.view(), |&x| x > 5).unwrap();
/// assert_eq!(rows.len(), 4);
/// assert_eq!(cols.len(), 4);
/// // The indices correspond to elements: 6, 7, 8, 9
/// ```
pub fn indices_where_2d<T, F>(array: ArrayView<T, Ix2>, condition: F) -> IndicesResult
where
    T: Clone,
    F: Fn(&T) -> bool,
{
    let (rows, cols) = (array.shape()[0], array.shape()[1]);

    // Build vectors of row and column indices where the condition is true
    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();

    for r in 0..rows {
        for c in 0..cols {
            if condition(&array[[r, c]]) {
                row_indices.push(r);
                col_indices.push(c);
            }
        }
    }

    // Convert the vectors to ndarray Arrays
    Ok((Array::from_vec(row_indices), Array::from_vec(col_indices)))
}

/// Return elements from a 2D array along an axis at specified indices
///
/// # Arguments
///
/// * `array` - The input 2D array
/// * `indices` - Indices to take along the specified axis
/// * `axis` - The axis along which to take values (0 for rows, 1 for columns)
///
/// # Returns
///
/// A 2D array with selected slices from the original array
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ndarray_ext::indexing::take_along_axis;
///
/// let a = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
/// let indices = array![0, 2];
///
/// // Take rows 0 and 2
/// let result = take_along_axis(a.view(), indices.view(), 0).unwrap();
/// assert_eq!(result.shape(), &[2, 3]);
/// assert_eq!(result[[0, 0]], 1);
/// assert_eq!(result[[0, 1]], 2);
/// assert_eq!(result[[0, 2]], 3);
/// assert_eq!(result[[1, 0]], 7);
/// assert_eq!(result[[1, 1]], 8);
/// assert_eq!(result[[1, 2]], 9);
/// ```
pub fn take_along_axis<T>(
    array: ArrayView<T, Ix2>,
    indices: ArrayView<usize, Ix1>,
    axis: usize,
) -> Result<Array<T, Ix2>, &'static str>
where
    T: Clone + Default,
{
    take_2d(array, indices, axis)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_boolean_mask_1d() {
        let a = array![1, 2, 3, 4, 5];
        let mask = array![true, false, true, false, true];

        let result = boolean_mask_1d(a.view(), mask.view()).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 3);
        assert_eq!(result[2], 5);
    }

    #[test]
    fn test_boolean_mask_2d() {
        let a = array![[1, 2, 3], [4, 5, 6]];
        let mask = array![[true, false, true], [false, true, false]];

        let result = boolean_mask_2d(a.view(), mask.view()).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 3);
        assert_eq!(result[2], 5);
    }

    #[test]
    fn test_take_1d() {
        let a = array![10, 20, 30, 40, 50];
        let indices = array![0, 2, 4];

        let result = take_1d(a.view(), indices.view()).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 10);
        assert_eq!(result[1], 30);
        assert_eq!(result[2], 50);
    }

    #[test]
    fn test_take_2d() {
        let a = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        let indices = array![0, 2];

        // Take along axis 0 (rows)
        let result = take_2d(a.view(), indices.view(), 0).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result[[0, 0]], 1);
        assert_eq!(result[[0, 1]], 2);
        assert_eq!(result[[0, 2]], 3);
        assert_eq!(result[[1, 0]], 7);
        assert_eq!(result[[1, 1]], 8);
        assert_eq!(result[[1, 2]], 9);

        // Take along axis 1 (columns)
        let result = take_2d(a.view(), indices.view(), 1).unwrap();
        assert_eq!(result.shape(), &[3, 2]);
        assert_eq!(result[[0, 0]], 1);
        assert_eq!(result[[0, 1]], 3);
        assert_eq!(result[[1, 0]], 4);
        assert_eq!(result[[1, 1]], 6);
        assert_eq!(result[[2, 0]], 7);
        assert_eq!(result[[2, 1]], 9);
    }

    #[test]
    fn test_fancy_index_2d() {
        let a = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        let row_indices = array![0, 2];
        let col_indices = array![0, 1];

        let result = fancy_index_2d(a.view(), row_indices.view(), col_indices.view()).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 8);
    }

    #[test]
    fn test_diagonal() {
        let a = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];

        // Main diagonal
        let main_diag = diagonal(a.view(), 0).unwrap();
        assert_eq!(main_diag.len(), 3);
        assert_eq!(main_diag[0], 1);
        assert_eq!(main_diag[1], 5);
        assert_eq!(main_diag[2], 9);

        // Upper diagonal
        let upper_diag = diagonal(a.view(), 1).unwrap();
        assert_eq!(upper_diag.len(), 2);
        assert_eq!(upper_diag[0], 2);
        assert_eq!(upper_diag[1], 6);

        // Lower diagonal
        let lower_diag = diagonal(a.view(), -1).unwrap();
        assert_eq!(lower_diag.len(), 2);
        assert_eq!(lower_diag[0], 4);
        assert_eq!(lower_diag[1], 8);
    }

    #[test]
    fn test_where_1d() {
        let a = array![1, 2, 3, 4, 5];

        let result = where_1d(a.view(), |&x| x > 3).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 4);
        assert_eq!(result[1], 5);
    }

    #[test]
    fn test_where_2d() {
        let a = array![[1, 2, 3], [4, 5, 6]];

        let result = where_2d(a.view(), |&x| x > 3).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 4);
        assert_eq!(result[1], 5);
        assert_eq!(result[2], 6);
    }

    #[test]
    fn test_indices_where_1d() {
        let a = array![10, 20, 30, 40, 50];

        let result = indices_where_1d(a.view(), |&x| x > 30).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 3);
        assert_eq!(result[1], 4);
    }

    #[test]
    fn test_indices_where_2d() {
        let a = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];

        let (rows, cols) = indices_where_2d(a.view(), |&x| x > 5).unwrap();
        assert_eq!(rows.len(), 4);
        assert_eq!(cols.len(), 4);

        // Verify that the indices correspond to the expected elements
        for (r, c) in rows.iter().zip(cols.iter()) {
            assert!(a[[*r, *c]] > 5);
        }
    }

    #[test]
    fn test_take_along_axis() {
        let a = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        let indices = array![0, 2];

        // Test along axis 0 (rows)
        let result = take_along_axis(a.view(), indices.view(), 0).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result[[0, 0]], 1);
        assert_eq!(result[[0, 1]], 2);
        assert_eq!(result[[0, 2]], 3);
        assert_eq!(result[[1, 0]], 7);
        assert_eq!(result[[1, 1]], 8);
        assert_eq!(result[[1, 2]], 9);
    }
}
