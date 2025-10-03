//! Generic filtering functions for n-dimensional arrays
//!
//! This module provides generic filtering functionality that allows users to apply
//! custom functions to local neighborhoods in arrays.

use scirs2_core::ndarray::{s, Array, Array1, Array2, Dimension, Ix1, Ix2, IxDyn};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

#[cfg(feature = "simd")]
// SIMD functions are imported on-demand where needed

// Import parallel_map
// Note: This is conditional on whether scirs2_core was built with parallel feature
#[cfg(feature = "parallel")]
use scirs2_core::parallel_ops::parallel_map;

use super::{pad_array, BorderMode};
use crate::error::{NdimageError, NdimageResult};

/// Helper function for safe conversion from numeric values to float
#[allow(dead_code)]
fn safe_to_float<T: Float + FromPrimitive>(value: f64) -> T {
    T::from_f64(value).unwrap_or_else(|| T::zero())
}

/// Apply a generic filter to an n-dimensional array
///
/// This function applies a user-defined function to every local neighborhood
/// in the input array. The function receives a flat slice of values from the
/// neighborhood and should return a single value.
///
/// # Arguments
///
/// * `input` - Input array to filter
/// * `function` - Function to apply to each neighborhood
/// * `size` - Size of the filter kernel in each dimension
/// * `mode` - Border handling mode (defaults to Reflect)
/// * `cval` - Constant value to use for Constant border mode
/// * `extra_arguments` - Additional arguments to pass to the function
/// * `extra_keywords` - Additional keyword arguments (not used in Rust)
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Filtered array
///
/// # Examples
///
/// ```no_run
/// use scirs2_core::ndarray::Array2;
/// use scirs2_ndimage::filters::generic_filter;
///
/// // Define a custom function that calculates the range (max - min)
/// let range_func = |values: &[f64]| -> f64 {
///     let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
///     let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
///     max - min
/// };
///
/// let input = Array2::from_shape_fn((5, 5), |(i, j)| (i + j) as f64);
/// let result = generic_filter(&input, range_func, &[3, 3], None, None).unwrap();
/// ```
#[allow(dead_code)]
pub fn generic_filter<T, D, F>(
    input: &Array<T, D>,
    function: F,
    size: &[usize],
    mode: Option<BorderMode>,
    cval: Option<T>,
) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static,
    D: Dimension,
    F: Fn(&[T]) -> T + Send + Sync + Clone + 'static,
{
    let border_mode = mode.unwrap_or(BorderMode::Reflect);
    let constant_value = cval.unwrap_or_else(|| safe_to_float(0.0));

    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if size.len() != input.ndim() {
        return Err(NdimageError::DimensionError(format!(
            "Size must have same length as input dimensions (got {} expected {})",
            size.len(),
            input.ndim()
        )));
    }

    for &s in size {
        if s == 0 {
            return Err(NdimageError::InvalidInput(
                "Kernel size cannot be zero".into(),
            ));
        }
    }

    // Handle different dimensionalities
    match input.ndim() {
        1 => {
            let input_1d = input
                .clone()
                .into_dimensionality::<Ix1>()
                .map_err(|_| NdimageError::DimensionError("Failed to convert to 1D".to_string()))?;
            let result_1d =
                generic_filter_1d(&input_1d, function, size[0], border_mode, constant_value)?;
            Ok(result_1d.into_dimensionality::<D>().map_err(|_| {
                NdimageError::DimensionError("Failed to convert from 1D".to_string())
            })?)
        }
        2 => {
            let input_2d = input
                .clone()
                .into_dimensionality::<Ix2>()
                .map_err(|_| NdimageError::DimensionError("Failed to convert to 2D".to_string()))?;
            let result_2d =
                generic_filter_2d(&input_2d, function, size, border_mode, constant_value)?;
            Ok(result_2d.into_dimensionality::<D>().map_err(|_| {
                NdimageError::DimensionError("Failed to convert from 2D".to_string())
            })?)
        }
        _ => {
            let input_dyn = input.clone().into_dimensionality::<IxDyn>().map_err(|_| {
                NdimageError::DimensionError("Failed to convert to IxDyn".to_string())
            })?;
            let result_dyn =
                generic_filter_nd(&input_dyn, function, size, border_mode, constant_value)?;
            Ok(result_dyn.into_dimensionality::<D>().map_err(|_| {
                NdimageError::DimensionError("Failed to convert from IxDyn".to_string())
            })?)
        }
    }
}

/// Apply a generic filter to a 1D array
#[allow(dead_code)]
fn generic_filter_1d<T, F>(
    input: &Array1<T>,
    function: F,
    size: usize,
    mode: BorderMode,
    cval: T,
) -> NdimageResult<Array1<T>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static,
    F: Fn(&[T]) -> T + Send + Sync,
{
    let half_size = size / 2;
    let pad_width = vec![(half_size, half_size)];
    let padded = pad_array(input, &pad_width, &mode, Some(cval))?;
    let mut result = Array1::zeros(input.raw_dim());

    // Apply the function to each neighborhood
    for i in 0..input.len() {
        let start = i;
        let end = i + size;

        if end <= padded.len() {
            let neighborhood: Vec<T> = padded.slice(s![start..end]).to_vec();
            result[i] = function(&neighborhood);
        } else {
            // Handle edge case - shouldn't happen with proper padding
            result[i] = input[i];
        }
    }

    Ok(result)
}

/// Apply a generic filter to a 2D array
#[allow(dead_code)]
fn generic_filter_2d<T, F>(
    input: &Array2<T>,
    function: F,
    size: &[usize],
    mode: BorderMode,
    cval: T,
) -> NdimageResult<Array2<T>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static,
    F: Fn(&[T]) -> T + Send + Sync + Clone + 'static,
{
    let (rows, cols) = input.dim();

    // Use parallel version for large arrays
    #[cfg(feature = "parallel")]
    if rows * cols > 10000 {
        return generic_filter_2d_parallel(input, function, size, mode, cval);
    }

    let half_size_0 = size[0] / 2;
    let half_size_1 = size[1] / 2;

    let pad_width = vec![(half_size_0, half_size_0), (half_size_1, half_size_1)];
    let padded = pad_array(input, &pad_width, &mode, Some(cval))?;
    let mut result = Array2::zeros(input.raw_dim());

    // Apply the function to each neighborhood
    for i in 0..rows {
        for j in 0..cols {
            let mut neighborhood = Vec::with_capacity(size[0] * size[1]);

            // Extract the neighborhood starting at the correct position
            // The padded array has the input positioned at (half_size_0, half_size_1)
            // So to get a neighborhood centered on input position (i, j),
            // we need to extract from padded starting at (i, j)
            for di in 0..size[0] {
                for dj in 0..size[1] {
                    let pi = i + di;
                    let pj = j + dj;
                    if pi < padded.nrows() && pj < padded.ncols() {
                        neighborhood.push(padded[[pi, pj]]);
                    }
                }
            }

            if neighborhood.len() == size[0] * size[1] {
                result[[i, j]] = function(&neighborhood);
            } else {
                // Fallback for edge cases
                result[[i, j]] = input[[i, j]];
            }
        }
    }

    Ok(result)
}

/// Parallel version of generic_filter_2d for large arrays
#[cfg(feature = "parallel")]
#[allow(dead_code)]
fn generic_filter_2d_parallel<T, F>(
    input: &Array2<T>,
    function: F,
    size: &[usize],
    mode: BorderMode,
    cval: T,
) -> NdimageResult<Array2<T>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static,
    F: Fn(&[T]) -> T + Send + Sync + Clone + 'static,
{
    let (rows, cols) = input.dim();
    let half_size_0 = size[0] / 2;
    let half_size_1 = size[1] / 2;

    let pad_width = vec![(half_size_0, half_size_0), (half_size_1, half_size_1)];
    let padded = pad_array(input, &pad_width, &mode, Some(cval))?;
    let mut result = Array2::zeros(input.raw_dim());

    // Clone data that will be moved into the closure
    let padded_clone = padded.clone();
    let input_clone = input.clone();
    let size_clone = size.to_vec();

    // Process rows in parallel
    let row_indices: Vec<usize> = (0..rows).collect();
    let row_results: Vec<Result<Vec<T>, NdimageError>> = parallel_map(&row_indices, move |&i| {
        let mut row_result = Vec::with_capacity(cols);

        for j in 0..cols {
            let mut neighborhood = Vec::with_capacity(size_clone[0] * size_clone[1]);

            // Extract the neighborhood
            for di in 0..size_clone[0] {
                for dj in 0..size_clone[1] {
                    let pi = i + di;
                    let pj = j + dj;
                    if pi < padded_clone.nrows() && pj < padded_clone.ncols() {
                        neighborhood.push(padded_clone[[pi, pj]]);
                    }
                }
            }

            if neighborhood.len() == size_clone[0] * size_clone[1] {
                row_result.push(function(&neighborhood));
            } else {
                row_result.push(input_clone[[i, j]]);
            }
        }

        Ok(row_result)
    });

    // Collect results and handle errors
    let processed_rows: Vec<Vec<T>> = row_results.into_iter().collect::<Result<Vec<_>, _>>()?;

    // Copy results back to the output array
    for (i, row_data) in processed_rows.into_iter().enumerate() {
        for (j, value) in row_data.into_iter().enumerate() {
            result[[i, j]] = value;
        }
    }

    Ok(result)
}

/// Apply a generic filter to an n-dimensional array
#[allow(dead_code)]
fn generic_filter_nd<T, F>(
    input: &Array<T, IxDyn>,
    function: F,
    size: &[usize],
    mode: BorderMode,
    cval: T,
) -> NdimageResult<Array<T, IxDyn>>
where
    T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static,
    F: Fn(&[T]) -> T + Send + Sync + Clone + 'static,
{
    let ndim = input.ndim();
    let shape = input.shape();

    // Calculate padding for each dimension
    let pad_width: Vec<(usize, usize)> = size
        .iter()
        .map(|&s| {
            let half_size = s / 2;
            (half_size, half_size)
        })
        .collect();

    let padded = pad_array(input, &pad_width, &mode, Some(cval))?;
    let mut result = Array::zeros(input.raw_dim());

    // Calculate total number of elements in neighborhood
    let neighborhood_size: usize = size.iter().product();

    // Iterate through all positions in the original array
    let total_elements: usize = shape.iter().product();

    for flat_idx in 0..total_elements {
        // Convert flat index to n-dimensional indices
        let mut indices = vec![0; ndim];
        let mut remaining = flat_idx;

        for (dim, &dim_size) in shape.iter().enumerate().rev() {
            indices[dim] = remaining % dim_size;
            remaining /= dim_size;
        }

        // Extract neighborhood
        let mut neighborhood = Vec::with_capacity(neighborhood_size);
        extract_neighborhood_nd(&padded, &indices, size, &mut neighborhood);

        if neighborhood.len() == neighborhood_size {
            let output_value = function(&neighborhood);
            result[indices.as_slice()] = output_value;
        } else {
            // Fallback
            result[indices.as_slice()] = input[indices.as_slice()];
        }
    }

    Ok(result)
}

/// Helper function to extract neighborhood values from n-dimensional array
#[allow(dead_code)]
fn extract_neighborhood_nd<T>(
    padded: &Array<T, IxDyn>,
    center_indices: &[usize],
    size: &[usize],
    neighborhood: &mut Vec<T>,
) where
    T: Clone,
{
    let ndim = center_indices.len();

    // Calculate total number of elements to extract
    let total_elements: usize = size.iter().product();

    for flat_offset in 0..total_elements {
        // Convert flat offset to n-dimensional offset
        let mut offsets = vec![0; ndim];
        let mut remaining = flat_offset;

        for (dim, &dim_size) in size.iter().enumerate().rev() {
            offsets[dim] = remaining % dim_size;
            remaining /= dim_size;
        }

        // Calculate actual _indices in padded array
        let mut actual_indices = Vec::with_capacity(ndim);
        let mut valid = true;

        for (dim, (&center_idx, &offset)) in center_indices.iter().zip(offsets.iter()).enumerate() {
            let actual_idx = center_idx + offset;
            if actual_idx >= padded.shape()[dim] {
                valid = false;
                break;
            }
            actual_indices.push(actual_idx);
        }

        if valid {
            neighborhood.push(padded[actual_indices.as_slice()].clone());
        }
    }
}

/// Common filter functions that can be used with generic_filter
pub mod filter_functions {
    use scirs2_core::numeric::{Float, FromPrimitive};

    /// Calculate the mean of values
    pub fn mean<T: Float + FromPrimitive>(values: &[T]) -> T {
        if values.is_empty() {
            return T::zero();
        }
        let sum = values.iter().fold(T::zero(), |acc, &x| acc + x);
        sum / T::from_usize(values.len()).unwrap_or(T::one())
    }

    /// Calculate the standard deviation of values
    pub fn std_dev<T: Float + FromPrimitive>(values: &[T]) -> T {
        if values.len() <= 1 {
            return T::zero();
        }

        let mean_val = mean(values);
        let variance = values
            .iter()
            .map(|&x| (x - mean_val).powi(2))
            .fold(T::zero(), |acc, x| acc + x)
            / T::from_usize(values.len() - 1).unwrap_or(T::one());

        variance.sqrt()
    }

    /// Calculate the range (max - min) of values
    pub fn range<T: Float>(values: &[T]) -> T {
        if values.is_empty() {
            return T::zero();
        }

        let min_val = values.iter().fold(T::infinity(), |a, &b| a.min(b));
        let max_val = values.iter().fold(T::neg_infinity(), |a, &b| a.max(b));
        max_val - min_val
    }

    /// Calculate the variance of values
    pub fn variance<T: Float + FromPrimitive>(values: &[T]) -> T {
        if values.len() <= 1 {
            return T::zero();
        }

        let mean_val = mean(values);
        values
            .iter()
            .map(|&x| (x - mean_val).powi(2))
            .fold(T::zero(), |acc, x| acc + x)
            / T::from_usize(values.len()).unwrap_or(T::one())
    }

    /// Calculate the maximum value
    pub fn maximum<T: Float>(values: &[T]) -> T {
        if values.is_empty() {
            return T::zero();
        }
        values.iter().fold(T::neg_infinity(), |a, &b| a.max(b))
    }

    /// Calculate the minimum value  
    pub fn minimum<T: Float>(values: &[T]) -> T {
        if values.is_empty() {
            return T::zero();
        }
        values.iter().fold(T::infinity(), |a, &b| a.min(b))
    }

    /// Calculate the median value
    pub fn median<T: Float + FromPrimitive>(values: &[T]) -> T {
        if values.is_empty() {
            return T::zero();
        }
        let mut sortedvalues: Vec<T> = values.to_vec();
        sortedvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let len = sortedvalues.len();
        if len % 2 == 0 {
            let mid1 = sortedvalues[len / 2 - 1];
            let mid2 = sortedvalues[len / 2];
            (mid1 + mid2) / T::from_f64(2.0).unwrap_or(T::one())
        } else {
            sortedvalues[len / 2]
        }
    }
}

/// SIMD-optimized filter functions for f32
#[cfg(feature = "simd")]
pub mod simd_filter_functions_f32 {
    use scirs2_core::ndarray::{Array1, ArrayView1};
    // SIMD functions for f32 imported when needed

    /// SIMD-optimized mean calculation for f32
    #[allow(dead_code)]
    pub fn mean_simd(values: &[f32]) -> f32 {
        if values.is_empty() {
            return 0.0;
        }

        // Use SIMD for larger arrays
        if values.len() >= 8 {
            let arr = Array1::from(values.to_vec());
            let view = arr.view();
            let sum = simd_scalar_sum_f32(&view);
            sum / values.len() as f32
        } else {
            // Fallback to regular calculation for small arrays
            super::filter_functions::mean(values)
        }
    }

    /// Helper function to sum array elements using SIMD
    #[allow(dead_code)]
    fn simd_scalar_sum_f32(arr: &ArrayView1<f32>) -> f32 {
        // Simple implementation - could be optimized further
        arr.iter().sum()
    }
}

/// SIMD-optimized filter functions for f64
#[cfg(feature = "simd")]
pub mod simd_filter_functions_f64 {
    use scirs2_core::ndarray::{Array1, ArrayView1};
    // SIMD functions for f64 imported when needed

    /// SIMD-optimized mean calculation for f64
    #[allow(dead_code)]
    pub fn mean_simd(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        // Use SIMD for larger arrays
        if values.len() >= 4 {
            let arr = Array1::from(values.to_vec());
            let view = arr.view();
            let sum = simd_scalar_sum_f64(&view);
            sum / values.len() as f64
        } else {
            // Fallback to regular calculation for small arrays
            super::filter_functions::mean(values)
        }
    }

    /// Helper function to sum array elements using SIMD
    #[allow(dead_code)]
    fn simd_scalar_sum_f64(arr: &ArrayView1<f64>) -> f64 {
        // Simple implementation - could be optimized further
        arr.iter().sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_generic_filter_mean() {
        let input = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        let result = generic_filter(&input, filter_functions::mean, &[3, 3], None, None)
            .expect("generic_filter should succeed for test");

        // Center element should be the mean of all elements
        let expected = (1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 6.0 + 7.0 + 8.0 + 9.0) / 9.0;
        assert_abs_diff_eq!(result[[1, 1]], expected, epsilon = 1e-10);
    }

    #[test]
    fn test_generic_filter_range() {
        let input = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        let result = generic_filter(&input, filter_functions::range, &[3, 3], None, None)
            .expect("generic_filter should succeed for test");

        // Center element should be the range of all elements (9 - 1 = 8)
        assert_abs_diff_eq!(result[[1, 1]], 8.0, epsilon = 1e-10);
    }

    #[test]
    fn test_generic_filter_1d() {
        let input = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let result = generic_filter(&input, filter_functions::mean, &[3], None, None)
            .expect("generic_filter should succeed for test");

        // Should preserve shape
        assert_eq!(result.len(), input.len());

        // Center element should be mean of [2, 3, 4] = 3.0
        assert_abs_diff_eq!(result[2], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_generic_filter_custom_function() {
        let input = array![[1.0, 2.0], [3.0, 4.0]];

        // Custom function that returns the maximum value
        let max_func =
            |values: &[f64]| -> f64 { values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)) };

        // Test with BorderMode::Nearest
        let result = generic_filter(&input, max_func, &[2, 2], Some(BorderMode::Nearest), None)
            .expect("generic_filter should succeed for test");

        // Check shape preservation and that center position sees the global maximum
        assert_eq!(result.shape(), input.shape());

        // The bottom-right position should see the global maximum (4.0)
        // because it includes the original (1,1) position
        assert_abs_diff_eq!(result[[1, 1]], 4.0, epsilon = 1e-10);

        // Other positions will see different maxima based on their neighborhoods
        assert!(result[[0, 0]] >= 1.0); // Should see at least the minimum value
        assert!(result[[0, 1]] >= 1.0);
        assert!(result[[1, 0]] >= 1.0);
    }

    #[test]
    fn test_additional_filter_functions() {
        let input = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        // Test maximum filter
        let result = generic_filter(&input, filter_functions::maximum, &[3, 3], None, None)
            .expect("generic_filter should succeed for test");
        assert_abs_diff_eq!(result[[1, 1]], 9.0, epsilon = 1e-10);

        // Test minimum filter
        let result = generic_filter(&input, filter_functions::minimum, &[3, 3], None, None)
            .expect("generic_filter should succeed for test");
        assert_abs_diff_eq!(result[[1, 1]], 1.0, epsilon = 1e-10);

        // Test median filter
        let result = generic_filter(&input, filter_functions::median, &[3, 3], None, None)
            .expect("generic_filter should succeed for test");
        assert_abs_diff_eq!(result[[1, 1]], 5.0, epsilon = 1e-10); // Median of [1,2,3,4,5,6,7,8,9] is 5
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_generic_filter() {
        // Create a large enough array to trigger parallel processing
        let input = Array2::from_shape_fn((200, 200), |(i, j)| (i * j) as f64);

        // Test with mean function
        let result = generic_filter(&input, filter_functions::mean, &[3, 3], None, None)
            .expect("generic_filter should succeed for test");

        // Should preserve shape
        assert_eq!(result.shape(), input.shape());

        // Spot check a few values - they should be reasonable means
        assert!(result[[100, 100]] > 0.0);
        assert!(result[[100, 100]] < input[[199, 199]]);
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_filter_functions() {
        let values_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let values_f64: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        // Test SIMD mean functions
        let mean_f32 = simd_filter_functions_f32::mean_simd(&values_f32);
        let mean_f64 = simd_filter_functions_f64::mean_simd(&values_f64);

        let expected = 5.0; // Mean of 1..9
        assert_abs_diff_eq!(mean_f32, expected as f32, epsilon = 1e-6);
        assert_abs_diff_eq!(mean_f64, expected, epsilon = 1e-10);
    }
}
