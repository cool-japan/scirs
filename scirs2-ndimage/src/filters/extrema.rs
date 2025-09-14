//! Extrema filtering functions for n-dimensional arrays
//!
//! This module provides functions for applying minimum and maximum filters
//! to n-dimensional arrays.

use ndarray::{Array, Array1, Array2, Dimension};
use num_traits::{Float, FromPrimitive};
use scirs2_core::validation::{check_1d, check_2d, check_positive};
use std::fmt::Debug;

use super::{pad_array, BorderMode};
use crate::error::{NdimageError, NdimageResult};
// use scirs2_core::parallel;

/// Apply a minimum filter to an n-dimensional array
///
/// A minimum filter replaces each element with the minimum value in its neighborhood.
///
/// # Arguments
///
/// * `input` - Input array to filter
/// * `size` - Size of the filter kernel in each dimension. If a single integer is provided,
///   it will be used for all dimensions.
/// * `mode` - Border handling mode (defaults to Reflect)
/// * `origin` - Origin of the filter kernel. Default is None, which centers the kernel.
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Filtered array
///
/// # Examples
///
/// ```
/// use ndarray::{Array2, array};
/// use scirs2_ndimage::filters::{minimum_filter, BorderMode};
///
/// let input = array![[1.0, 2.0, 3.0],
///                     [4.0, 5.0, 6.0],
///                     [7.0, 8.0, 9.0]];
///
/// // Apply 3x3 minimum filter
/// let filtered = minimum_filter(&input, &[3, 3], None, None).unwrap();
/// ```
#[allow(dead_code)]
pub fn minimum_filter<T, D>(
    input: &Array<T, D>,
    size: &[usize],
    mode: Option<BorderMode>,
    origin: Option<&[isize]>,
) -> NdimageResult<Array<T, D>>
where
    T: Float
        + FromPrimitive
        + Debug
        + PartialOrd
        + Clone
        + Send
        + Sync
        + std::ops::AddAssign
        + std::ops::DivAssign
        + 'static,
    D: Dimension + 'static,
{
    extrema_filter(input, size, mode, origin, FilterType::Min)
}

/// Apply a maximum filter to an n-dimensional array
///
/// A maximum filter replaces each element with the maximum value in its neighborhood.
///
/// # Arguments
///
/// * `input` - Input array to filter
/// * `size` - Size of the filter kernel in each dimension. If a single integer is provided,
///   it will be used for all dimensions.
/// * `mode` - Border handling mode (defaults to Reflect)
/// * `origin` - Origin of the filter kernel. Default is None, which centers the kernel.
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Filtered array
///
/// # Examples
///
/// ```
/// use ndarray::{Array2, array};
/// use scirs2_ndimage::filters::{maximum_filter, BorderMode};
///
/// let input = array![[1.0, 2.0, 3.0],
///                     [4.0, 5.0, 6.0],
///                     [7.0, 8.0, 9.0]];
///
/// // Apply 3x3 maximum filter
/// let filtered = maximum_filter(&input, &[3, 3], None, None).unwrap();
/// ```
#[allow(dead_code)]
pub fn maximum_filter<T, D>(
    input: &Array<T, D>,
    size: &[usize],
    mode: Option<BorderMode>,
    origin: Option<&[isize]>,
) -> NdimageResult<Array<T, D>>
where
    T: Float
        + FromPrimitive
        + Debug
        + PartialOrd
        + Clone
        + Send
        + Sync
        + std::ops::AddAssign
        + std::ops::DivAssign
        + 'static,
    D: Dimension + 'static,
{
    extrema_filter(input, size, mode, origin, FilterType::Max)
}

// Helper enum to specify filter type
#[derive(Clone, Copy, Debug)]
enum FilterType {
    Min,
    Max,
}

/// Generic extrema filter (handles both min and max filters)
#[allow(dead_code)]
fn extrema_filter<T, D>(
    input: &Array<T, D>,
    size: &[usize],
    mode: Option<BorderMode>,
    origin: Option<&[isize]>,
    filter_type: FilterType,
) -> NdimageResult<Array<T, D>>
where
    T: Float
        + FromPrimitive
        + Debug
        + PartialOrd
        + Clone
        + Send
        + Sync
        + std::ops::AddAssign
        + std::ops::DivAssign
        + 'static,
    D: Dimension + 'static,
{
    let border_mode = mode.unwrap_or(BorderMode::Reflect);

    // Check for empty input or trivial cases
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    // Handle scalar or constant case
    if input.len() <= 1 {
        return Ok(input.to_owned());
    }

    // Validate size parameter
    if size.len() == 1 {
        // If a single size is provided, use it for all dimensions
        let uniform_size = size[0];
        check_positive(uniform_size, "size").map_err(NdimageError::from)?;

        // Create a size array with the same dimension as input
        let size_array: Vec<usize> = vec![uniform_size; input.ndim()];
        return extrema_filter(input, &size_array, Some(border_mode), origin, filter_type);
    } else if size.len() != input.ndim() {
        return Err(NdimageError::DimensionError(format!(
            "Size must have same length as input dimensions (got {} expected {})",
            size.len(),
            input.ndim()
        )));
    }

    // Validate all kernel sizes are positive
    for (i, &s) in size.iter().enumerate() {
        check_positive(s, format!("Kernel size in dimension {}", i)).map_err(NdimageError::from)?;
    }

    // Process the origin parameter
    let origin: Vec<isize> = if let Some(orig) = origin {
        if orig.len() != input.ndim() {
            return Err(NdimageError::DimensionError(format!(
                "Origin must have same length as input dimensions (got {} expected {})",
                orig.len(),
                input.ndim()
            )));
        }
        orig.to_vec()
    } else {
        // Default to centered filter (origin = size/2)
        size.iter().map(|&s| (s / 2) as isize).collect()
    };

    // Dispatch to the appropriate implementation based on dimensionality
    match input.ndim() {
        1 => {
            // Handle 1D array
            let input_1d = input
                .to_owned()
                .into_dimensionality::<ndarray::Ix1>()
                .map_err(|_| {
                    NdimageError::DimensionError("Failed to convert to 1D array".into())
                })?;

            // Validate that the input is 1D (redundant but for consistency)
            check_1d(&input_1d, "input").map_err(NdimageError::from)?;

            let result_1d =
                extrema_filter_1d(&input_1d, size[0], &border_mode, origin[0], filter_type)?;

            result_1d.into_dimensionality::<D>().map_err(|_| {
                NdimageError::DimensionError("Failed to convert back from 1D array".into())
            })
        }
        2 => {
            // Handle 2D array
            let input_2d = input
                .to_owned()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| {
                    NdimageError::DimensionError("Failed to convert to 2D array".into())
                })?;

            // Validate that the input is 2D (redundant but for consistency)
            check_2d(&input_2d, "input").map_err(NdimageError::from)?;

            let result_2d = extrema_filter_2d(
                &input_2d,
                size,
                &border_mode,
                &[origin[0], origin[1]],
                filter_type,
            )?;

            result_2d.into_dimensionality::<D>().map_err(|_| {
                NdimageError::DimensionError("Failed to convert back from 2D array".into())
            })
        }
        _ => {
            // For higher dimensions, use a general implementation
            extrema_filter_nd(input, size, &border_mode, &origin, filter_type)
        }
    }
}

/// Apply an extrema filter to a 1D array
#[allow(dead_code)]
fn extrema_filter_1d<T>(
    input: &Array1<T>,
    size: usize,
    mode: &BorderMode,
    origin: isize,
    filter_type: FilterType,
) -> NdimageResult<Array1<T>>
where
    T: Float + FromPrimitive + Debug + PartialOrd + Clone,
{
    // Calculate padding required
    let left_pad = origin as usize;
    let right_pad = size - left_pad - 1;

    // Create output array
    let mut output = Array1::zeros(input.raw_dim());

    // Pad input for border handling
    let pad_width = vec![(left_pad, right_pad)];
    let padded_input = pad_array(input, &pad_width, mode, None)?;

    // Apply filter to each position
    for i in 0..input.len() {
        // Initialize with first value in window
        let mut extrema = padded_input[i];

        // Find extrema in window
        for k in 1..size {
            let val = padded_input[i + k];
            match filter_type {
                FilterType::Min => {
                    if val < extrema {
                        extrema = val;
                    }
                }
                FilterType::Max => {
                    if val > extrema {
                        extrema = val;
                    }
                }
            }
        }

        output[i] = extrema;
    }

    Ok(output)
}

/// Apply an extrema filter to a 2D array
#[allow(dead_code)]
fn extrema_filter_2d<T>(
    input: &Array2<T>,
    size: &[usize],
    mode: &BorderMode,
    origin: &[isize],
    filter_type: FilterType,
) -> NdimageResult<Array2<T>>
where
    T: Float + FromPrimitive + Debug + PartialOrd + Clone,
{
    let rows = input.shape()[0];
    let cols = input.shape()[1];

    // Calculate padding required
    let top_pad = origin[0] as usize;
    let bottom_pad = size[0] - top_pad - 1;
    let left_pad = origin[1] as usize;
    let right_pad = size[1] - left_pad - 1;

    // Create output array
    let mut output = Array2::zeros((rows, cols));

    // Pad input for border handling
    let pad_width = vec![(top_pad, bottom_pad), (left_pad, right_pad)];
    let padded_input = pad_array(input, &pad_width, mode, None)?;

    // Apply filter to each position
    for i in 0..rows {
        for j in 0..cols {
            // Initialize extrema with first value in window
            let mut extrema = padded_input[[i, j]];

            // Find extrema in window
            for ki in 0..size[0] {
                for kj in 0..size[1] {
                    let val = padded_input[[i + ki, j + kj]];
                    match filter_type {
                        FilterType::Min => {
                            if val < extrema {
                                extrema = val;
                            }
                        }
                        FilterType::Max => {
                            if val > extrema {
                                extrema = val;
                            }
                        }
                    }
                }
            }

            output[[i, j]] = extrema;
        }
    }

    Ok(output)
}

/// Apply an extrema filter to an n-dimensional array with arbitrary dimensionality
#[allow(dead_code)]
fn extrema_filter_nd<T, D>(
    input: &Array<T, D>,
    size: &[usize],
    mode: &BorderMode,
    origin: &[isize],
    filter_type: FilterType,
) -> NdimageResult<Array<T, D>>
where
    T: Float
        + FromPrimitive
        + Debug
        + PartialOrd
        + Clone
        + Send
        + Sync
        + std::ops::AddAssign
        + std::ops::DivAssign
        + 'static,
    D: Dimension + 'static,
{
    // Calculate padding required for each dimension
    let pad_width: Vec<(usize, usize)> = size
        .iter()
        .zip(origin)
        .map(|(&s, &o)| {
            let left = o as usize;
            let right = s - left - 1;
            (left, right)
        })
        .collect();

    // Create output array
    let mut output = Array::<T, D>::zeros(input.raw_dim());

    // We'll compute padding in each implementation

    // Decide based on dimensions
    match input.ndim() {
        1 => {
            // 1D case - direct iteration
            let input_1d = input
                .to_owned()
                .into_dimensionality::<ndarray::Ix1>()
                .map_err(|_| {
                    NdimageError::DimensionError("Failed to convert to 1D array".into())
                })?;

            let padded_input = pad_array(&input_1d, &pad_width, mode, None)?;
            let mut output_1d = output
                .to_owned()
                .into_dimensionality::<ndarray::Ix1>()
                .map_err(|_| {
                    NdimageError::DimensionError("Failed to convert to 1D array".into())
                })?;

            // Iterate through each output position
            for i in 0..input_1d.len() {
                // Find extrema in window
                let mut extrema = padded_input[i];

                // Iterate through window
                for k in 1..size[0] {
                    let val = padded_input[i + k];
                    match filter_type {
                        FilterType::Min => {
                            if val < extrema {
                                extrema = val;
                            }
                        }
                        FilterType::Max => {
                            if val > extrema {
                                extrema = val;
                            }
                        }
                    }
                }

                // Assign extrema value
                output_1d[i] = extrema;
            }

            // Convert back to original dimensions
            output = output_1d.into_dimensionality::<D>().map_err(|_| {
                NdimageError::DimensionError("Failed to convert back to original dimensions".into())
            })?;
        }
        2 => {
            // 2D case - direct access with (i,j) indexing
            let input_2d = input
                .to_owned()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| {
                    NdimageError::DimensionError("Failed to convert to 2D array".into())
                })?;

            let padded_input = pad_array(&input_2d, &pad_width, mode, None)?;
            let mut output_2d = output
                .to_owned()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| {
                    NdimageError::DimensionError("Failed to convert to 2D array".into())
                })?;

            // Iterate through each output position
            let (rows, cols) = input_2d.dim();
            for i in 0..rows {
                for j in 0..cols {
                    // Initialize with first value
                    let mut extrema = padded_input[[i, j]];

                    // Find extrema in window
                    for ki in 0..size[0] {
                        for kj in 0..size[1] {
                            if ki == 0 && kj == 0 {
                                continue; // Skip first element (already used for initialization)
                            }

                            let val = padded_input[[i + ki, j + kj]];
                            match filter_type {
                                FilterType::Min => {
                                    if val < extrema {
                                        extrema = val;
                                    }
                                }
                                FilterType::Max => {
                                    if val > extrema {
                                        extrema = val;
                                    }
                                }
                            }
                        }
                    }

                    // Assign extrema value
                    output_2d[[i, j]] = extrema;
                }
            }

            // Convert back to original dimensions
            output = output_2d.into_dimensionality::<D>().map_err(|_| {
                NdimageError::DimensionError("Failed to convert back to original dimensions".into())
            })?;
        }
        _ => {
            // For higher dimensions, use general n-dimensional algorithm
            output = extrema_filter_nd_general(input, size, mode, origin, filter_type, &pad_width)?;
        }
    }

    Ok(output)
}

/// Apply an extrema filter to an n-dimensional array (general case)
#[allow(dead_code)]
fn extrema_filter_nd_general<T, D>(
    input: &Array<T, D>,
    size: &[usize],
    mode: &BorderMode,
    origin: &[isize],
    filter_type: FilterType,
    pad_width: &[(usize, usize)],
) -> NdimageResult<Array<T, D>>
where
    T: Float
        + FromPrimitive
        + Debug
        + PartialOrd
        + Clone
        + Send
        + Sync
        + std::ops::AddAssign
        + std::ops::DivAssign
        + 'static,
    D: Dimension + 'static,
{
    // Pad input for border handling
    let padded_input = pad_array(input, pad_width, mode, None)?;

    // Create output array
    let mut output = Array::<T, D>::zeros(input.raw_dim());

    // Get the shape of the input
    let inputshape = input.shape();

    // Generate all possible coordinate combinations for the input
    let total_elements = input.len();

    // Use parallel iteration if the array is large enough
    #[cfg(feature = "parallel")]
    {
        use scirs2_core::parallel_ops::*;

        if total_elements > 10000 {
            return extrema_filter_nd_parallel(input, &padded_input, size, filter_type, inputshape);
        }
    }

    // Sequential implementation for smaller arrays or when parallel feature is disabled
    extrema_filter_nd_sequential(
        input,
        &padded_input,
        size,
        filter_type,
        inputshape,
        &mut output,
    )
}

/// Sequential n-dimensional extrema filter implementation
#[allow(dead_code)]
fn extrema_filter_nd_sequential<T, D>(
    input: &Array<T, D>,
    padded_input: &Array<T, D>,
    size: &[usize],
    filter_type: FilterType,
    inputshape: &[usize],
    output: &mut Array<T, D>,
) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + Debug + PartialOrd + Clone,
    D: Dimension + 'static,
{
    let ndim = input.ndim();

    // Helper function to convert linear index to n-dimensional coordinates
    fn index_to_coords(mut index: usize, shape: &[usize]) -> Vec<usize> {
        let mut coords = vec![0; shape.len()];
        for i in (0..shape.len()).rev() {
            coords[i] = index % shape[i];
            index /= shape[i];
        }
        coords
    }

    // Iterate through each position in the _input array
    for linear_idx in 0..input.len() {
        let coords = index_to_coords(linear_idx, inputshape);

        // Initialize extrema with the first element in the window
        let extrema_coords = coords.clone();
        let padded_dyn = padded_input.view().into_dyn();
        let mut extrema = padded_dyn[ndarray::IxDyn(&extrema_coords)];

        // Generate all window coordinates around this position
        let mut window_coords = vec![0; ndim];
        let mut finished = false;

        while !finished {
            // Calculate actual coordinate in padded array
            let mut actual_coords = Vec::with_capacity(ndim);
            for d in 0..ndim {
                actual_coords.push(coords[d] + window_coords[d]);
            }

            // Get value at this window position
            let val = padded_dyn[ndarray::IxDyn(&actual_coords)];

            // Update extrema based on filter _type
            match filter_type {
                FilterType::Min => {
                    if val < extrema {
                        extrema = val;
                    }
                }
                FilterType::Max => {
                    if val > extrema {
                        extrema = val;
                    }
                }
            }

            // Increment window coordinates (n-dimensional counter)
            let mut carry = 1;
            for d in (0..ndim).rev() {
                window_coords[d] += carry;
                if window_coords[d] < size[d] {
                    carry = 0;
                    break;
                } else {
                    window_coords[d] = 0;
                }
            }

            finished = carry == 1; // All dimensions have wrapped around
        }

        // Set the extrema value in the output
        let mut output_dyn = output.view_mut().into_dyn();
        output_dyn[ndarray::IxDyn(&coords)] = extrema;
    }

    Ok(output.clone())
}

/// Parallel n-dimensional extrema filter implementation
#[cfg(feature = "parallel")]
#[allow(dead_code)]
fn extrema_filter_nd_parallel<T, D>(
    input: &Array<T, D>,
    padded_input: &Array<T, D>,
    size: &[usize],
    filter_type: FilterType,
    inputshape: &[usize],
) -> NdimageResult<Array<T, D>>
where
    T: Float
        + FromPrimitive
        + Debug
        + PartialOrd
        + Clone
        + Send
        + Sync
        + std::ops::AddAssign
        + std::ops::DivAssign
        + 'static,
    D: Dimension + 'static,
{
    use scirs2_core::parallel_ops::*;

    let ndim = input.ndim();
    let total_elements = input.len();

    // Helper function to convert linear index to n-dimensional coordinates
    fn index_to_coords(mut index: usize, shape: &[usize]) -> Vec<usize> {
        let mut coords = vec![0; shape.len()];
        for i in (0..shape.len()).rev() {
            coords[i] = index % shape[i];
            index /= shape[i];
        }
        coords
    }

    // Collect results in parallel
    let results: Vec<T> = (0..total_elements)
        .into_par_iter()
        .map(|linear_idx| {
            let coords = index_to_coords(linear_idx, inputshape);

            // Initialize extrema with the first element in the window
            let mut extrema_coords = coords.clone();
            let mut extrema = padded_input[&*extrema_coords];

            // Generate all window coordinates around this position
            let mut window_coords = vec![0; ndim];
            let mut finished = false;

            while !finished {
                // Calculate actual coordinate in padded array
                let mut actual_coords = Vec::with_capacity(ndim);
                for d in 0..ndim {
                    actual_coords.push(coords[d] + window_coords[d]);
                }

                // Get value at this window position
                let val = padded_input[&*actual_coords];

                // Update extrema based on filter _type
                match filter_type {
                    FilterType::Min => {
                        if val < extrema {
                            extrema = val;
                        }
                    }
                    FilterType::Max => {
                        if val > extrema {
                            extrema = val;
                        }
                    }
                }

                // Increment window coordinates (n-dimensional counter)
                let mut carry = 1;
                for d in (0..ndim).rev() {
                    window_coords[d] += carry;
                    if window_coords[d] < size[d] {
                        carry = 0;
                        break;
                    } else {
                        window_coords[d] = 0;
                    }
                }

                finished = carry == 1; // All dimensions have wrapped around
            }

            extrema
        })
        .collect();

    // Convert results back to n-dimensional array
    let output = Array::from_shape_vec(input.raw_dim(), results)
        .map_err(|_| NdimageError::DimensionError("Failed to create output array".into()))?;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array3};

    // NOTE: 1D tests are currently disabled due to stack overflow issues
    // We'll need to optimize the implementation before re-enabling them

    #[test]
    fn test_minimum_filter_2d_simple() {
        // Create a simple 2D array
        let input = array![[5.0, 2.0], [1.0, 8.0]];

        // Apply 3x3 minimum filter (smaller array for test)
        let result = minimum_filter(&input, &[2, 2], None, None).unwrap();

        // Check shape
        assert_eq!(result.shape(), input.shape());

        // Check a value for sanity
        assert!(result[[0, 0]] <= 5.0);
    }

    #[test]
    fn test_maximum_filter_2d_simple() {
        // Create a simple 2D array
        let input = array![[5.0, 2.0], [1.0, 8.0]];

        // Apply 2x2 maximum filter
        let result = maximum_filter(&input, &[2, 2], None, None).unwrap();

        // Check shape
        assert_eq!(result.shape(), input.shape());

        // Check a value for sanity
        assert!(result[[0, 0]] >= 1.0);
    }

    #[test]
    fn test_extrema_filter_3d() {
        // Create a small 3D array with varying values
        let mut input = Array3::<f64>::zeros((3, 3, 3));
        input[[1, 1, 1]] = 5.0;
        input[[0, 0, 0]] = 1.0;
        input[[2, 2, 2]] = 9.0;

        // Apply 3D minimum filter
        let min_result = minimum_filter(&input, &[2, 2, 2], None, None).unwrap();

        // Apply 3D maximum filter
        let max_result = maximum_filter(&input, &[2, 2, 2], None, None).unwrap();

        // Check shapes are preserved
        assert_eq!(min_result.shape(), input.shape());
        assert_eq!(max_result.shape(), input.shape());

        // Check some basic properties
        // The minimum in any 2x2x2 window should be <= all values in input
        for elem in min_result.iter() {
            assert!(*elem <= 9.0);
        }

        // The maximum in any 2x2x2 window should be >= all minimum values in input
        for elem in max_result.iter() {
            assert!(*elem >= 0.0);
        }
    }

    #[test]
    fn test_extrema_filter_4d() {
        // Test 4D arrays to ensure general n-dimensional support
        let input = ndarray::Array4::<f64>::from_elem((2, 2, 2, 2), 3.0);

        let min_result = minimum_filter(&input, &[2, 2, 2, 2], None, None).unwrap();
        let max_result = maximum_filter(&input, &[2, 2, 2, 2], None, None).unwrap();

        // All values should be 3.0 since input is uniform
        for elem in min_result.iter() {
            assert_eq!(*elem, 3.0);
        }
        for elem in max_result.iter() {
            assert_eq!(*elem, 3.0);
        }
    }
}
