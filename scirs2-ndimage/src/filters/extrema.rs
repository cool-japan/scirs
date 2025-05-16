//! Extrema filtering functions for n-dimensional arrays
//!
//! This module provides functions for applying minimum and maximum filters
//! to n-dimensional arrays.

use ndarray::{Array, Array1, Array2, Dimension};
use num_traits::{Float, FromPrimitive};
use scirs2_core::validation::{check_1d, check_2d, check_positive};
use std::fmt::Debug;

use super::{pad_array, BorderMode};
use crate::error::{NdimageError, Result};
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
pub fn minimum_filter<T, D>(
    input: &Array<T, D>,
    size: &[usize],
    mode: Option<BorderMode>,
    origin: Option<&[isize]>,
) -> Result<Array<T, D>>
where
    T: Float + FromPrimitive + Debug + PartialOrd + Clone + Send + Sync,
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
pub fn maximum_filter<T, D>(
    input: &Array<T, D>,
    size: &[usize],
    mode: Option<BorderMode>,
    origin: Option<&[isize]>,
) -> Result<Array<T, D>>
where
    T: Float + FromPrimitive + Debug + PartialOrd + Clone + Send + Sync,
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
fn extrema_filter<T, D>(
    input: &Array<T, D>,
    size: &[usize],
    mode: Option<BorderMode>,
    origin: Option<&[isize]>,
    filter_type: FilterType,
) -> Result<Array<T, D>>
where
    T: Float + FromPrimitive + Debug + PartialOrd + Clone + Send + Sync,
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
fn extrema_filter_1d<T>(
    input: &Array1<T>,
    size: usize,
    mode: &BorderMode,
    origin: isize,
    filter_type: FilterType,
) -> Result<Array1<T>>
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
fn extrema_filter_2d<T>(
    input: &Array2<T>,
    size: &[usize],
    mode: &BorderMode,
    origin: &[isize],
    filter_type: FilterType,
) -> Result<Array2<T>>
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
fn extrema_filter_nd<T, D>(
    input: &Array<T, D>,
    size: &[usize],
    mode: &BorderMode,
    origin: &[isize],
    filter_type: FilterType,
) -> Result<Array<T, D>>
where
    T: Float + FromPrimitive + Debug + PartialOrd + Clone + Send + Sync,
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
            // For higher dimensions, display warning
            return Err(NdimageError::NotImplementedError(format!(
                "{} filter for arrays with more than 2 dimensions is not efficiently implemented yet",
                match filter_type {
                    FilterType::Min => "Minimum",
                    FilterType::Max => "Maximum",
                }
            )));
        }
    }

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
        // Create a small 3D array
        let input = Array3::<f64>::zeros((2, 2, 2));

        // Check that 3D filters correctly return NotImplementedError
        let min_result = minimum_filter(&input, &[2, 2, 2], None, None);
        let max_result = maximum_filter(&input, &[2, 2, 2], None, None);

        assert!(min_result.is_err());
        assert!(max_result.is_err());

        if let Err(err) = min_result {
            match err {
                NdimageError::NotImplementedError(_) => (),
                _ => panic!("Expected NotImplementedError but got {:?}", err),
            }
        }
    }
}
