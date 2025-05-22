// Index dtype handling utilities for sparse arrays
//
// This module provides utilities for handling index dtypes in sparse arrays,
// similar to SciPy's index dtype handling in the sparse module.

use ndarray::{Array1, ArrayView1};
use num_traits::PrimInt;
use std::cmp::max;

use crate::error::{SparseError, SparseResult};

/// Determine the appropriate index dtype based on array size.
///
/// This function selects the smallest appropriate index dtype for a sparse array
/// based on the maximum element count and maximum value in any index arrays.
///
/// # Parameters
///
/// * `shape`: The shape of the sparse array (rows, columns)
/// * `idx_arrays`: Optional array views to check for maximum values
///
/// # Returns
///
/// A string representing the recommended index dtype ("i32", "i64", or "usize")
///
/// # Examples
///
/// ```
/// use ndarray::Array1;
/// use scirs2_sparse::index_dtype::get_index_dtype;
///
/// // Small array, i32 is sufficient
/// let shape = (100, 100);
/// let indices = vec![0, 5, 10, 20];
/// let idx_array = Array1::from_vec(indices);
/// let dtype = get_index_dtype(shape, &[idx_array.view()]);
/// assert_eq!(dtype, "i32");
///
/// // Larger array, might need i64
/// let large_shape = (2_000_000_000, 2_000_000_000);
/// let dtype_large = get_index_dtype(large_shape, &[]);
/// assert_eq!(dtype_large, "i64");
/// ```
pub fn get_index_dtype(shape: (usize, usize), idx_arrays: &[ArrayView1<usize>]) -> &'static str {
    let (rows, cols) = shape;

    // Maximum index value that could be needed (product of dimensions)
    let theoretical_max = rows.saturating_mul(cols);

    // Find the maximum value in any of the index arrays
    let observed_max = if idx_arrays.is_empty() {
        0
    } else {
        idx_arrays
            .iter()
            .flat_map(|arr| arr.iter())
            .fold(0, |acc, &x| max(acc, x))
    };

    // Use the larger of the theoretical and observed maximums
    let max_value = max(theoretical_max, observed_max);

    // Choose dtype based on max value
    if max_value <= i32::MAX as usize {
        "i32"
    } else if max_value <= i64::MAX as usize {
        "i64"
    } else {
        "usize"
    }
}

/// Safely cast index arrays to the specified type.
///
/// This function converts index arrays to the specified dtype,
/// ensuring that no values are lost during conversion.
///
/// # Type Parameters
///
/// * `T`: The target integer type to cast to
///
/// # Parameters
///
/// * `arrays`: A slice of array views to convert
///
/// # Returns
///
/// A `SparseResult` containing a vector of the converted arrays,
/// or an error if any values would be lost in the conversion.
///
/// # Examples
///
/// ```
/// use ndarray::Array1;
/// use scirs2_sparse::index_dtype::safely_cast_index_arrays;
///
/// // Valid conversion (all values fit in i32)
/// let indices = vec![0, 5, 10, 20];
/// let idx_array = Array1::from_vec(indices);
/// let result = safely_cast_index_arrays::<i32>(&[idx_array.view()]);
/// assert!(result.is_ok());
///
/// // Invalid conversion (value too large for i8)
/// let large_indices = vec![0, 5, 10, 200];
/// let large_array = Array1::from_vec(large_indices);
/// let result = safely_cast_index_arrays::<i8>(&[large_array.view()]);
/// assert!(result.is_err());
/// ```
pub fn safely_cast_index_arrays<T>(arrays: &[ArrayView1<usize>]) -> SparseResult<Vec<Array1<T>>>
where
    T: PrimInt + 'static + TryFrom<usize>,
    <T as TryFrom<usize>>::Error: std::fmt::Debug,
{
    let mut result = Vec::with_capacity(arrays.len());

    for array in arrays {
        let mut converted = Array1::uninit(array.len());

        for (i, &val) in array.iter().enumerate() {
            match T::try_from(val) {
                Ok(converted_val) => {
                    // Set the value and mark it as initialized
                    unsafe {
                        converted.uget_mut(i).write(converted_val);
                    }
                }
                Err(_) => {
                    return Err(SparseError::IndexCastOverflow {
                        value: val,
                        target_type: std::any::type_name::<T>(),
                    });
                }
            }
        }

        // This is safe because we've initialized all elements
        // by writing converted values to them
        let safe_array = unsafe { converted.assume_init() };
        result.push(safe_array);
    }

    Ok(result)
}

/// Check if an array can be safely cast to the target type.
///
/// # Type Parameters
///
/// * `T`: The target integer type to check against
///
/// # Parameters
///
/// * `array`: The array view to check
///
/// # Returns
///
/// `true` if all values in the array can be represented in the target type,
/// `false` otherwise.
pub fn can_cast_safely<T>(array: ArrayView1<usize>) -> bool
where
    T: PrimInt + 'static + TryFrom<usize>,
    <T as TryFrom<usize>>::Error: std::fmt::Debug,
{
    for &val in array.iter() {
        if T::try_from(val).is_err() {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_get_index_dtype_small() {
        let shape = (100, 100);
        let dtype = get_index_dtype(shape, &[]);
        assert_eq!(dtype, "i32");
    }

    #[test]
    fn test_get_index_dtype_medium() {
        let shape = (50_000, 50_000);
        let dtype = get_index_dtype(shape, &[]);
        assert_eq!(dtype, "i64");
    }

    #[test]
    fn test_get_index_dtype_large() {
        let shape = (usize::MAX / 2, 3);
        let dtype = get_index_dtype(shape, &[]);
        assert_eq!(dtype, "usize");
    }

    #[test]
    fn test_get_index_dtype_with_arrays() {
        let indices1 = Array1::from_vec(vec![0, 10, 20, 30]);
        let indices2 = Array1::from_vec(vec![5, 15, 25, 1000]);

        let dtype = get_index_dtype((100, 100), &[indices1.view(), indices2.view()]);
        assert_eq!(dtype, "i32");
    }

    #[test]
    fn test_get_index_dtype_with_large_values() {
        let indices = Array1::from_vec(vec![0, i32::MAX as usize + 1]);

        let dtype = get_index_dtype((100, 100), &[indices.view()]);
        assert_eq!(dtype, "i64");
    }

    #[test]
    fn test_safely_cast_valid() {
        let indices = Array1::from_vec(vec![0, 5, 10, 100]);

        let result = safely_cast_index_arrays::<i32>(&[indices.view()]);
        assert!(result.is_ok());

        let arrays = result.unwrap();
        assert_eq!(arrays.len(), 1);
        assert_eq!(arrays[0].len(), 4);
        assert_eq!(arrays[0][2], 10);
    }

    #[test]
    fn test_safely_cast_multiple() {
        let indices1 = Array1::from_vec(vec![0, 5, 10]);
        let indices2 = Array1::from_vec(vec![1, 2, 3, 4]);

        let result = safely_cast_index_arrays::<i32>(&[indices1.view(), indices2.view()]);
        assert!(result.is_ok());

        let arrays = result.unwrap();
        assert_eq!(arrays.len(), 2);
        assert_eq!(arrays[0].len(), 3);
        assert_eq!(arrays[1].len(), 4);
    }

    #[test]
    fn test_safely_cast_invalid() {
        let indices = Array1::from_vec(vec![0, 5, 10, 200]);

        let result = safely_cast_index_arrays::<i8>(&[indices.view()]);
        assert!(result.is_err());

        match result {
            Err(SparseError::IndexCastOverflow { value, target_type }) => {
                assert_eq!(value, 200);
                assert_eq!(target_type, "i8");
            }
            _ => panic!("Expected IndexCastOverflow error"),
        }
    }

    #[test]
    fn test_can_cast_safely() {
        let small_indices = Array1::from_vec(vec![0, 5, 10, 20]);
        assert!(can_cast_safely::<i8>(small_indices.view()));

        let large_indices = Array1::from_vec(vec![0, 5, 10, 200]);
        assert!(!can_cast_safely::<i8>(large_indices.view()));
    }
}
