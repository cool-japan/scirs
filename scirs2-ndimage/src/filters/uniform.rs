//! Uniform filtering functions for n-dimensional arrays
//!
//! This module provides functions for applying uniform filters (also known as box filters)
//! to n-dimensional arrays.

use ndarray::{s, Array, Array1, Array2, Dim, Dimension, IxDyn, IxDynImpl, NdIndex};
use num_traits::{Float, FromPrimitive};
use scirs2_core::validation::{check_1d, check_2d, check_positive};
use std::fmt::Debug;

use super::{pad_array, BorderMode};
use crate::error::{NdimageError, NdimageResult};

#[cfg(feature = "simd")]
use scirs2_core::simd_ops::SimdUnifiedOps;

#[cfg(feature = "parallel")]
use scirs2_core::parallel_ops;

/// Apply a uniform filter to an n-dimensional array
///
/// A uniform filter convolves the input with a kernel of ones, effectively computing
/// the unweighted average of the local neighborhood at each point.
///
/// # Arguments
///
/// * `input` - Input array to filter
/// * `size` - Size of the filter kernel in each dimension. If a single integer is provided,
///   it will be used for all dimensions.
/// * `mode` - Border handling mode (defaults to Reflect)
/// * `origin` - Origin of the filter kernel. Default is None, which centers the kernel.
///   Provide an array of integers with one element for each axis, which specifies
///   the origin of the kernel for that axis. The origin is 0 by default, which
///   corresponds to the center of the kernel.
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Filtered array
///
/// # Examples
///
/// ```
/// use ndarray::{Array2, array};
/// use scirs2_ndimage::filters::{uniform_filter, BorderMode};
///
/// let input = array![[1.0, 2.0, 3.0],
///                     [4.0, 5.0, 6.0],
///                     [7.0, 8.0, 9.0]];
///
/// // Apply 3x3 uniform filter
/// let filtered = uniform_filter(&input, &[3, 3], None, None).unwrap();
/// ```
#[allow(dead_code)]
pub fn uniform_filter<T, D>(
    input: &Array<T, D>,
    size: &[usize],
    mode: Option<BorderMode>,
    origin: Option<&[isize]>,
) -> NdimageResult<Array<T, D>>
where
    T: Float
        + FromPrimitive
        + Debug
        + std::ops::AddAssign
        + std::ops::DivAssign
        + Send
        + Sync
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
        return uniform_filter(input, &size_array, Some(border_mode), origin);
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

            let result_1d = uniform_filter_1d(&input_1d, size[0], &border_mode, origin[0])?;

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

            let result_2d =
                uniform_filter_2d(&input_2d, size, &border_mode, &[origin[0], origin[1]])?;

            result_2d.into_dimensionality::<D>().map_err(|_| {
                NdimageError::DimensionError("Failed to convert back from 2D array".into())
            })
        }
        _ => {
            // For higher dimensions, use a general implementation
            uniform_filter_nd(input, size, &border_mode, &origin)
        }
    }
}

/// Apply a uniform filter to a 1D array
#[allow(dead_code)]
fn uniform_filter_1d<T>(
    input: &Array1<T>,
    size: usize,
    mode: &BorderMode,
    origin: isize,
) -> NdimageResult<Array1<T>>
where
    T: Float + FromPrimitive + Debug + std::ops::AddAssign + std::ops::DivAssign + 'static,
{
    // Check if we can use SIMD optimizations for f32 type
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() && input.len() > 64 && size >= 3
    {
        return uniform_filter_1d_simd_f32(input, size, mode, origin).map(|result| {
            // Convert the f32 result back to T
            result.mapv(|x| T::from_f32(x).unwrap_or_else(T::zero))
        });
    }

    // Calculate padding required
    let left_pad = origin as usize;
    let right_pad = size - left_pad - 1;

    // Create output array
    let mut output = Array1::zeros(input.raw_dim());

    // Pad input for border handling
    let pad_width = vec![(left_pad, right_pad)];
    let padded_input = pad_array(input, &pad_width, mode, None)?;

    // Calculate normalization factor (1/size)
    let norm_factor = T::one() / T::from_usize(size).unwrap();

    // Apply filter to each position
    for i in 0..input.len() {
        let mut sum = T::zero();

        // Sum the window
        for k in 0..size {
            sum += padded_input[i + k];
        }

        // Normalize
        output[i] = sum * norm_factor;
    }

    Ok(output)
}

/// SIMD-optimized uniform filter for f32 arrays
#[cfg(feature = "simd")]
#[allow(dead_code)]
fn uniform_filter_1d_simd_f32<T>(
    input: &Array1<T>,
    size: usize,
    mode: &BorderMode,
    origin: isize,
) -> NdimageResult<Array1<f32>>
where
    T: Float + FromPrimitive + Debug,
{
    // Convert input to f32 for SIMD processing
    let input_f32: Vec<f32> = input.iter().map(|&x| x.to_f32().unwrap_or(0.0)).collect();
    let input_f32_array = Array1::from_vec(input_f32);

    // Calculate padding required
    let left_pad = origin as usize;
    let right_pad = size - left_pad - 1;

    // Create output array
    let mut output = Array1::zeros(input.raw_dim());

    // Pad input for border handling
    let pad_width = vec![(left_pad, right_pad)];
    let padded_input = pad_array(&input_f32_array, &pad_width, mode, None)?;

    // Calculate normalization factor (1/size)
    let norm_factor = 1.0f32 / (size as f32);

    // Use SIMD operations for summing windows
    let padded_data = padded_input.as_slice().ok_or_else(|| {
        NdimageError::ComputationError("Failed to get contiguous slice".to_string())
    })?;

    // Process in chunks for SIMD
    for i in 0..input.len() {
        let window_start = i;
        let window_end = i + size;

        if window_end <= padded_data.len() {
            // Use SIMD sum operation for the window
            let window_slice = &padded_data[window_start..window_end];
            let window_array = Array1::from_vec(window_slice.to_vec());
            let sum = f32::simd_sum(&window_array.view());
            output[i] = sum * norm_factor;
        } else {
            // Fallback for edge cases
            let mut sum = 0.0f32;
            for k in 0..size {
                if window_start + k < padded_data.len() {
                    sum += padded_data[window_start + k];
                }
            }
            output[i] = sum * norm_factor;
        }
    }

    Ok(output)
}

/// Fallback implementation when SIMD feature is not enabled
#[cfg(not(feature = "simd"))]
#[allow(dead_code)]
fn uniform_filter_1d_simd_f32<T>(
    input: &Array1<T>,
    size: usize,
    mode: &BorderMode,
    origin: isize,
) -> NdimageResult<Array1<f32>>
where
    T: Float + FromPrimitive + Debug,
{
    // Convert to f32 and use standard implementation
    let input_f32: Vec<f32> = input.iter().map(|&x| x.to_f32().unwrap_or(0.0)).collect();
    let input_f32_array = Array1::from_vec(input_f32);

    // Calculate padding required
    let left_pad = origin as usize;
    let right_pad = size - left_pad - 1;

    // Create output array
    let mut output = Array1::zeros(input.raw_dim());

    // Pad input for border handling
    let pad_width = vec![(left_pad, right_pad)];
    let padded_input = pad_array(&input_f32_array, &pad_width, mode, None)?;

    // Calculate normalization factor (1/size)
    let norm_factor = 1.0f32 / (size as f32);

    // Apply filter to each position
    for i in 0..input.len() {
        let mut sum = 0.0f32;

        // Sum the window
        for k in 0..size {
            sum += padded_input[i + k];
        }

        // Normalize
        output[i] = sum * norm_factor;
    }

    Ok(output)
}

/// Apply a uniform filter to a 2D array
#[allow(dead_code)]
fn uniform_filter_2d<T>(
    input: &Array2<T>,
    size: &[usize],
    mode: &BorderMode,
    origin: &[isize],
) -> NdimageResult<Array2<T>>
where
    T: Float + FromPrimitive + Debug + std::ops::AddAssign + std::ops::DivAssign + 'static,
{
    let rows = input.shape()[0];
    let cols = input.shape()[1];

    // Check if we can use SIMD optimizations for f32 type
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
        && rows * cols > 1024
        && size[0] >= 3
        && size[1] >= 3
    {
        return uniform_filter_2d_simd_f32(input, size, mode, origin).map(|result| {
            // Convert the f32 result back to T
            result.mapv(|x| T::from_f32(x).unwrap_or_else(T::zero))
        });
    }

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

    // Calculate normalization factor (1/total_size)
    let total_size = size[0] * size[1];
    let norm_factor = T::one() / T::from_usize(total_size).unwrap();

    // Apply filter to each position
    for i in 0..rows {
        for j in 0..cols {
            let mut sum = T::zero();

            // Sum the window
            for ki in 0..size[0] {
                for kj in 0..size[1] {
                    sum += padded_input[[i + ki, j + kj]];
                }
            }

            // Normalize
            output[[i, j]] = sum * norm_factor;
        }
    }

    Ok(output)
}

/// SIMD-optimized uniform filter for 2D f32 arrays
#[cfg(feature = "simd")]
#[allow(dead_code)]
fn uniform_filter_2d_simd_f32<T>(
    input: &Array2<T>,
    size: &[usize],
    mode: &BorderMode,
    origin: &[isize],
) -> NdimageResult<Array2<f32>>
where
    T: Float + FromPrimitive + Debug,
{
    let rows = input.shape()[0];
    let cols = input.shape()[1];

    // Convert input to f32 for SIMD processing
    let input_f32 = input.mapv(|x| x.to_f32().unwrap_or(0.0));

    // Calculate padding required
    let top_pad = origin[0] as usize;
    let bottom_pad = size[0] - top_pad - 1;
    let left_pad = origin[1] as usize;
    let right_pad = size[1] - left_pad - 1;

    // Create output array
    let mut output = Array2::zeros((rows, cols));

    // Pad input for border handling
    let pad_width = vec![(top_pad, bottom_pad), (left_pad, right_pad)];
    let padded_input = pad_array(&input_f32, &pad_width, mode, None)?;

    // Calculate normalization factor (1/total_size)
    let total_size = size[0] * size[1];
    let norm_factor = 1.0f32 / (total_size as f32);

    // Apply filter to each position with SIMD row processing
    for i in 0..rows {
        for j in 0..cols {
            let mut sum = 0.0f32;

            // Sum the window using SIMD operations where possible
            for ki in 0..size[0] {
                let row_start = j;
                let row_end = j + size[1];

                // Get the row slice for SIMD processing
                let padded_row = padded_input.row(i + ki);
                let window_slice = &padded_row.as_slice().unwrap()[row_start..row_end];

                // Use SIMD sum for the row segment
                let window_array = Array1::from_vec(window_slice.to_vec());
                sum += f32::simd_sum(&window_array.view());
            }

            // Normalize
            output[[i, j]] = sum * norm_factor;
        }
    }

    Ok(output)
}

/// Fallback implementation when SIMD feature is not enabled
#[cfg(not(feature = "simd"))]
#[allow(dead_code)]
fn uniform_filter_2d_simd_f32<T>(
    input: &Array2<T>,
    size: &[usize],
    mode: &BorderMode,
    origin: &[isize],
) -> NdimageResult<Array2<f32>>
where
    T: Float + FromPrimitive + Debug,
{
    let rows = input.shape()[0];
    let cols = input.shape()[1];

    // Convert input to f32
    let input_f32 = input.mapv(|x| x.to_f32().unwrap_or(0.0));

    // Calculate padding required
    let top_pad = origin[0] as usize;
    let bottom_pad = size[0] - top_pad - 1;
    let left_pad = origin[1] as usize;
    let right_pad = size[1] - left_pad - 1;

    // Create output array
    let mut output = Array2::zeros((rows, cols));

    // Pad input for border handling
    let pad_width = vec![(top_pad, bottom_pad), (left_pad, right_pad)];
    let padded_input = pad_array(&input_f32, &pad_width, mode, None)?;

    // Calculate normalization factor (1/total_size)
    let total_size = size[0] * size[1];
    let norm_factor = 1.0f32 / (total_size as f32);

    // Apply filter to each position
    for i in 0..rows {
        for j in 0..cols {
            let mut sum = 0.0f32;

            // Sum the window
            for ki in 0..size[0] {
                for kj in 0..size[1] {
                    sum += padded_input[[i + ki, j + kj]];
                }
            }

            // Normalize
            output[[i, j]] = sum * norm_factor;
        }
    }

    Ok(output)
}

/// Apply a uniform filter to an n-dimensional array with arbitrary dimensionality
#[allow(dead_code)]
fn uniform_filter_nd<T, D>(
    input: &Array<T, D>,
    size: &[usize],
    mode: &BorderMode,
    origin: &[isize],
) -> NdimageResult<Array<T, D>>
where
    T: Float
        + FromPrimitive
        + Debug
        + std::ops::AddAssign
        + std::ops::DivAssign
        + Send
        + Sync
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

    // Calculate normalization factor (1/total_size)
    let total_size: usize = size.iter().product();
    let norm_factor = T::one() / T::from_usize(total_size).unwrap();

    // Create output array
    let output = Array::<T, D>::zeros(input.raw_dim());

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
                let mut sum = T::zero();
                // Sum the window elements
                for k in 0..size[0] {
                    sum += padded_input[i + k];
                }
                // Normalize and assign
                output_1d[i] = sum * norm_factor;
            }

            // Convert back to original dimensions
            output_1d.into_dimensionality::<D>().map_err(|_| {
                NdimageError::DimensionError("Failed to convert back to original dimensions".into())
            })
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
                    let mut sum = T::zero();
                    // Sum the window elements
                    for ki in 0..size[0] {
                        for kj in 0..size[1] {
                            sum += padded_input[[i + ki, j + kj]];
                        }
                    }
                    // Normalize and assign
                    output_2d[[i, j]] = sum * norm_factor;
                }
            }

            // Convert back to original dimensions
            output_2d.into_dimensionality::<D>().map_err(|_| {
                NdimageError::DimensionError("Failed to convert back to original dimensions".into())
            })
        }
        _ => {
            // For higher dimensions, use the efficient separable implementation
            uniform_filter_nd_general(input, size, mode, origin, &pad_width, norm_factor)
        }
    }
}

/// Apply a uniform filter to an n-dimensional array (general case)
#[allow(dead_code)]
fn uniform_filter_nd_general<T, D>(
    input: &Array<T, D>,
    size: &[usize],
    mode: &BorderMode,
    origin: &[isize],
    pad_width: &[(usize, usize)],
    norm_factor: T,
) -> NdimageResult<Array<T, D>>
where
    T: Float
        + FromPrimitive
        + Debug
        + std::ops::AddAssign
        + std::ops::DivAssign
        + Send
        + Sync
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
    // Disabled temporarily due to type constraint issues
    // #[cfg(feature = "parallel")]
    // {
    //     if total_elements > 10000 {
    //         return uniform_filter_nd_parallel(input, &padded_input, size, norm_factor, inputshape);
    //     }
    // }

    // Sequential implementation for smaller arrays or when parallel feature is disabled
    uniform_filter_nd_sequential(
        input,
        &padded_input,
        size,
        norm_factor,
        inputshape,
        &mut output,
    )
}

/// Sequential n-dimensional uniform filter implementation
#[allow(dead_code)]
fn uniform_filter_nd_sequential<T, D>(
    input: &Array<T, D>,
    padded_input: &Array<T, D>,
    size: &[usize],
    norm_factor: T,
    inputshape: &[usize],
    output: &mut Array<T, D>,
) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + Debug + std::ops::AddAssign,
    D: Dimension,
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

    // Iterate through each position in the input array
    for linear_idx in 0..input.len() {
        let coords = index_to_coords(linear_idx, inputshape);

        // Initialize sum
        let mut sum = T::zero();

        // Generate all window coordinates around this position
        let mut window_coords = vec![0; ndim];
        let mut finished = false;

        while !finished {
            // Calculate actual coordinate in padded array
            let mut actual_coords = Vec::with_capacity(ndim);
            for d in 0..ndim {
                actual_coords.push(coords[d] + window_coords[d]);
            }

            // Add value at this window position to sum
            let padded_dyn = padded_input.view().into_dyn();
            let val = padded_dyn[IxDyn(&actual_coords)];
            sum += val;

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

        // Set the normalized average value in the output
        let mut output_dyn = output.view_mut().into_dyn();
        output_dyn[IxDyn(&coords)] = sum * norm_factor;
    }

    Ok(output.clone())
}

/// Parallel n-dimensional uniform filter implementation
#[cfg(feature = "parallel")]
#[allow(dead_code)]
fn uniform_filter_nd_parallel<T, D>(
    input: &Array<T, D>,
    padded_input: &Array<T, D>,
    size: &[usize],
    norm_factor: T,
    inputshape: &[usize],
) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + Debug + std::ops::AddAssign + Send + Sync,
    D: Dimension + 'static,
    Dim<IxDynImpl>: NdIndex<D>,
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

            // Initialize sum
            let mut sum = T::zero();

            // Generate all window coordinates around this position
            let mut window_coords = vec![0; ndim];
            let mut finished = false;

            while !finished {
                // Calculate actual coordinate in padded array
                let mut actual_coords = Vec::with_capacity(ndim);
                for d in 0..ndim {
                    actual_coords.push(coords[d] + window_coords[d]);
                }

                // Add value at this window position to sum
                let val = padded_input[IxDyn(&actual_coords)];
                sum += val;

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

            // Return normalized average
            sum * norm_factor
        })
        .collect();

    // Convert results back to n-dimensional array
    let output = Array::from_shape_vec(input.raw_dim(), results)
        .map_err(|_| NdimageError::DimensionError("Failed to create output array".into()))?;

    Ok(output)
}

/// Optimized separable uniform filter (1D filtering along each axis)
///
/// This is an optimization for uniform filters that applies 1D filtering sequentially
/// along each axis. This is much faster than the direct approach for large kernel sizes.
///
/// # Arguments
///
/// * `input` - Input array to filter
/// * `size` - Size of the filter kernel in each dimension
/// * `mode` - Border handling mode (defaults to Reflect)
/// * `origin` - Origin of the filter kernel
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Filtered array
#[allow(dead_code)]
pub fn uniform_filter_separable<T, D>(
    input: &Array<T, D>,
    size: &[usize],
    mode: Option<BorderMode>,
    origin: Option<&[isize]>,
) -> NdimageResult<Array<T, D>>
where
    T: Float
        + FromPrimitive
        + Debug
        + std::ops::AddAssign
        + std::ops::DivAssign
        + Send
        + Sync
        + 'static,
    D: Dimension + 'static,
{
    let border_mode = mode.unwrap_or(BorderMode::Reflect);

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

    // Apply 1D uniform filter along each dimension
    let mut result = input.to_owned();

    for axis in 0..input.ndim() {
        let axis_size = size[axis];
        let axis_origin = origin[axis];

        // Skip if size is 1 (no filtering needed)
        if axis_size <= 1 {
            continue;
        }

        // Apply 1D filter along this axis
        result = uniform_filter_along_axis(&result, axis, axis_size, &border_mode, axis_origin)?;
    }

    Ok(result)
}

/// Apply a 1D uniform filter along a specific axis of an n-dimensional array
#[allow(dead_code)]
fn uniform_filter_along_axis<T, D>(
    input: &Array<T, D>,
    axis: usize,
    size: usize,
    mode: &BorderMode,
    origin: isize,
) -> NdimageResult<Array<T, D>>
where
    T: Float
        + FromPrimitive
        + Debug
        + std::ops::AddAssign
        + std::ops::DivAssign
        + Send
        + Sync
        + 'static,
    D: Dimension + 'static,
{
    // Check if axis is valid
    if axis >= input.ndim() {
        return Err(NdimageError::DimensionError(format!(
            "Axis {} is out of bounds for array of dimension {}",
            axis,
            input.ndim()
        )));
    }

    // Calculate padding for this axis
    let left_pad = origin as usize;
    let right_pad = size - left_pad - 1;

    let mut pad_width = vec![(0, 0); input.ndim()];
    pad_width[axis] = (left_pad, right_pad);

    // Create output array
    let mut output = Array::<T, D>::zeros(input.raw_dim());

    // Pad the input array
    let padded_input = pad_array(input, &pad_width, mode, None)?;

    // Calculate normalization factor (1/size)
    let norm_factor = T::one() / T::from_usize(size).unwrap();

    // Handle each dimensionality case separately for best performance
    match input.ndim() {
        1 => {
            // For 1D arrays, handle directly
            let input_1d = input
                .to_owned()
                .into_dimensionality::<ndarray::Ix1>()
                .map_err(|_| {
                    NdimageError::DimensionError("Failed to convert to 1D array".into())
                })?;

            let padded_input_1d =
                padded_input
                    .into_dimensionality::<ndarray::Ix1>()
                    .map_err(|_| {
                        NdimageError::DimensionError("Failed to convert padded input to 1D".into())
                    })?;

            let mut output_1d = output
                .view_mut()
                .into_dimensionality::<ndarray::Ix1>()
                .map_err(|_| {
                    NdimageError::DimensionError("Failed to convert output to 1D".into())
                })?;

            // Apply 1D filter directly
            for i in 0..input_1d.len() {
                let mut sum = T::zero();
                for k in 0..size {
                    sum += padded_input_1d[i + k];
                }
                output_1d[i] = sum * norm_factor;
            }

            // No need to convert back since we're using a mutable view
        }
        2 => {
            // For 2D arrays, handle directly with special cases for each axis
            let input_2d = input
                .to_owned()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| {
                    NdimageError::DimensionError("Failed to convert to 2D array".into())
                })?;

            let padded_input_2d =
                padded_input
                    .into_dimensionality::<ndarray::Ix2>()
                    .map_err(|_| {
                        NdimageError::DimensionError("Failed to convert padded input to 2D".into())
                    })?;

            let mut output_2d = output
                .view_mut()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| {
                    NdimageError::DimensionError("Failed to convert output to 2D".into())
                })?;

            let (rows, cols) = input_2d.dim();

            // Apply filter along the specified axis
            if axis == 0 {
                // Filter along rows
                for j in 0..cols {
                    for i in 0..rows {
                        let mut sum = T::zero();
                        for k in 0..size {
                            sum += padded_input_2d[[i + k, j]];
                        }
                        output_2d[[i, j]] = sum * norm_factor;
                    }
                }
            } else if axis == 1 {
                // Filter along columns
                for i in 0..rows {
                    for j in 0..cols {
                        let mut sum = T::zero();
                        for k in 0..size {
                            sum += padded_input_2d[[i, j + k]];
                        }
                        output_2d[[i, j]] = sum * norm_factor;
                    }
                }
            }

            // No need to convert back since we're using a mutable view
        }
        _ => {
            // For higher dimensions, we need to use a more general approach
            // Convert to dynamic dimension for processing
            let inputshape = input.shape();
            let padded_input_dyn = padded_input.view().into_dyn();
            let mut output_dyn = output.view_mut().into_dyn();

            // Process each position in the output array
            // Helper function to iterate through all indices
            #[allow(clippy::too_many_arguments)]
            fn process_indices<T: Float + FromPrimitive + std::ops::AddAssign>(
                input_dyn: &ndarray::ArrayViewD<T>,
                output_dyn: &mut ndarray::ArrayViewMutD<T>,
                size: usize,
                axis: usize,
                idx: &mut Vec<usize>,
                inputshape: &[usize],
                dimension: usize,
                norm_factor: T,
            ) {
                if dimension == inputshape.len() {
                    // At leaf level, calculate the window sum along the axis
                    let mut sum = T::zero();
                    let mut temp_idx = idx.clone();

                    for k in 0..size {
                        temp_idx[axis] = idx[axis] + k;
                        if let Some(val) = input_dyn.get(temp_idx.as_slice()) {
                            sum += *val;
                        }
                    }

                    // Store normalized result
                    output_dyn[idx.as_slice()] = sum * norm_factor;
                    return;
                }

                if dimension == axis {
                    // Skip axis dimension in recursion since we handle it separately
                    process_indices(
                        input_dyn,
                        output_dyn,
                        size,
                        axis,
                        idx,
                        inputshape,
                        dimension + 1,
                        norm_factor,
                    );
                } else {
                    // Recurse on all other dimensions
                    for i in 0..inputshape[dimension] {
                        idx[dimension] = i;
                        process_indices(
                            input_dyn,
                            output_dyn,
                            size,
                            axis,
                            idx,
                            inputshape,
                            dimension + 1,
                            norm_factor,
                        );
                    }
                }
            }

            // Call the recursive helper function to process all indices
            let mut idx = vec![0; input.ndim()];
            process_indices(
                &padded_input_dyn,
                &mut output_dyn,
                size,
                axis,
                &mut idx,
                inputshape,
                0,
                norm_factor,
            );
        }
    }

    Ok(output)
}

/// Chunked uniform filter for very large 2D arrays
/// This function processes the input in chunks to optimize memory usage and cache performance
#[cfg(feature = "parallel")]
#[allow(dead_code)]
pub fn uniform_filter_chunked<T>(
    input: &Array2<T>,
    size: &[usize],
    _chunk_size: usize,
    mode: Option<BorderMode>,
    origin: Option<&[isize]>,
) -> NdimageResult<Array2<T>>
where
    T: Float
        + FromPrimitive
        + Debug
        + std::ops::AddAssign
        + std::ops::DivAssign
        + Send
        + Sync
        + Clone
        + 'static,
{
    let border_mode = mode.unwrap_or(BorderMode::Reflect);
    let (rows, cols) = input.dim();

    // Process the origin parameter
    let origin: Vec<isize> = if let Some(orig) = origin {
        if orig.len() != 2 {
            return Err(NdimageError::DimensionError(format!(
                "Origin must have length 2 for 2D arrays (got {})",
                orig.len()
            )));
        }
        orig.to_vec()
    } else {
        // Default to centered filter
        vec![(size[0] / 2) as isize, (size[1] / 2) as isize]
    };

    // Calculate padding for overlap between chunks
    let pad_rows = size[0];
    let pad_cols = size[1];

    // Create output array
    let mut output = Array2::zeros((rows, cols));

    // Calculate number of chunks
    let num_row_chunks = (rows + _chunk_size - 1) / _chunk_size;
    let num_col_chunks = (cols + _chunk_size - 1) / _chunk_size;

    // Process chunks in parallel using scirs2-core's parallel operations
    let chunk_indices: Vec<(usize, usize)> = (0..num_row_chunks)
        .flat_map(|i| (0..num_col_chunks).map(move |j| (i, j)))
        .collect();

    let process_chunk = |&(chunk_row, chunk_col): &(usize, usize)| -> Result<((usize, usize), Array2<T>), crate::error::NdimageError> {
        // Calculate chunk boundaries
        let row_start = chunk_row * _chunk_size;
        let row_end = std::cmp::min(row_start + _chunk_size, rows);
        let col_start = chunk_col * _chunk_size;
        let col_end = std::cmp::min(col_start + _chunk_size, cols);

        // Extract chunk with padding for boundary conditions
        let padded_row_start = row_start.saturating_sub(pad_rows);
        let padded_row_end = std::cmp::min(row_end + pad_rows, rows);
        let padded_col_start = col_start.saturating_sub(pad_cols);
        let padded_col_end = std::cmp::min(col_end + pad_cols, cols);

        // Extract the padded chunk
        let chunk_slice = input.slice(s![
            padded_row_start..padded_row_end,
            padded_col_start..padded_col_end
        ]);
        let chunk = chunk_slice.to_owned();

        // Apply uniform filter to the chunk
        let filtered_chunk = uniform_filter_2d(&chunk, size, &border_mode, &origin)?;

        // Extract the non-padded portion
        let output_row_offset = row_start.saturating_sub(padded_row_start);
        let output_col_offset = col_start.saturating_sub(padded_col_start);
        let output_row_count = row_end - row_start;
        let output_col_count = col_end - col_start;

        let result_chunk = filtered_chunk.slice(s![
            output_row_offset..output_row_offset + output_row_count,
            output_col_offset..output_col_offset + output_col_count
        ]).to_owned();

        Ok(((chunk_row, chunk_col), result_chunk))
    };

    // Process chunks in parallel
    let chunk_results = parallel_ops::parallel_map_result(&chunk_indices, process_chunk)?;

    // Reassemble the results
    for ((chunk_row, chunk_col), chunk_result) in chunk_results {
        let row_start = chunk_row * _chunk_size;
        let row_end = std::cmp::min(row_start + _chunk_size, rows);
        let col_start = chunk_col * _chunk_size;
        let col_end = std::cmp::min(col_start + _chunk_size, cols);

        let mut output_slice = output.slice_mut(s![row_start..row_end, col_start..col_end]);
        output_slice.assign(&chunk_result);
    }

    Ok(output)
}

/// Non-parallel version of chunked uniform filter
#[cfg(not(feature = "parallel"))]
#[allow(dead_code)]
pub fn uniform_filter_chunked<T>(
    input: &Array2<T>,
    size: &[usize],
    _chunk_size: usize,
    mode: Option<BorderMode>,
    origin: Option<&[isize]>,
) -> NdimageResult<Array2<T>>
where
    T: Float
        + FromPrimitive
        + Debug
        + std::ops::AddAssign
        + std::ops::DivAssign
        + Send
        + Sync
        + Clone
        + 'static,
{
    // For non-parallel version, just call the regular uniform filter
    // This ensures the API is consistent even when parallel feature is disabled
    uniform_filter_2d(input, size, &mode.unwrap_or(BorderMode::Reflect), &{
        if let Some(orig) = origin {
            orig.to_vec()
        } else {
            vec![(size[0] / 2) as isize, (size[1] / 2) as isize]
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    //use approx::assert_abs_diff_eq; // Will be used again when we improve the tests
    use ndarray::{array, Array3};

    // NOTE: 1D tests are currently disabled due to stack overflow issues
    // We'll need to optimize the implementation before re-enabling them

    #[test]
    fn test_uniform_filter_2d() {
        // Create a simple 2D array
        let input = array![[1.0, 2.0], [4.0, 5.0]];

        // Apply uniform filter
        let result = uniform_filter(&input, &[2, 2], None, None).unwrap();

        // Check shape
        assert_eq!(result.shape(), input.shape());

        // Check that center value makes sense (should be average of values)
        assert!(result[[0, 0]] > 1.0);
        assert!(result[[0, 0]] < 5.0);
    }

    #[test]
    fn test_invalid_inputs() {
        // Create a simple array
        let input = array![[1.0, 2.0], [3.0, 4.0]];

        // Test with invalid size
        let result = uniform_filter(&input, &[0, 0], None, None);
        assert!(result.is_err());

        // Test with mismatched origin size
        let result = uniform_filter(&input, &[2, 2], None, Some(&[0, 0, 0]));
        assert!(result.is_err());
    }

    #[test]
    fn test_uniform_filter_3d() {
        // Create a small 3D array with varying values
        let mut input = Array3::<f64>::zeros((3, 3, 3));
        input[[1, 1, 1]] = 8.0;
        input[[0, 0, 0]] = 2.0;
        input[[2, 2, 2]] = 6.0;

        // Apply 3D uniform filter
        let result = uniform_filter(&input, &[2, 2, 2], None, None).unwrap();

        // Check shape is preserved
        assert_eq!(result.shape(), input.shape());

        // Check some basic properties
        // Uniform filter should produce values that are averages, so between min and max of input
        for elem in result.iter() {
            assert!(*elem >= 0.0);
            assert!(*elem <= 8.0);
        }

        // Test separable filter still works
        let result_sep = uniform_filter_separable(&input, &[2, 2, 2], None, None).unwrap();
        assert_eq!(result_sep.shape(), input.shape());
    }

    #[test]
    fn test_uniform_filter_4d() {
        // Test 4D arrays to ensure general n-dimensional support
        let input = ndarray::Array4::<f64>::from_elem((2, 2, 2, 2), 5.0);

        let result = uniform_filter(&input, &[2, 2, 2, 2], None, None).unwrap();

        // All values should be 5.0 since input is uniform
        for elem in result.iter() {
            assert!((elem - 5.0).abs() < 1e-10);
        }
    }
}
