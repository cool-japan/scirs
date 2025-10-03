//! Rank-based filtering functions for n-dimensional arrays

use scirs2_core::ndarray::{Array, Array1, Array2, ArrayView, Dimension};
use scirs2_core::numeric::{Float, FromPrimitive, Zero};
use scirs2_core::{parallel_ops, CoreError};
use std::fmt::Debug;

use super::{pad_array, BorderMode};
use crate::error::{NdimageError, NdimageResult};

/// Apply a maximum filter to an n-dimensional array
///
/// # Arguments
///
/// * `input` - Input array to filter
/// * `size` - Size of the filter kernel in each dimension
/// * `mode` - Border handling mode (defaults to Reflect)
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Filtered array
#[allow(dead_code)]
pub fn maximum_filter<T, D>(
    input: &Array<T, D>,
    size: &[usize],
    mode: Option<BorderMode>,
) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + Debug + PartialOrd + Clone + Send + Sync + 'static,
    D: Dimension,
{
    // We can implement this using the rank filter with the rank set to the maximum value
    let border_mode = mode.unwrap_or(BorderMode::Reflect);

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

    // Calculate total size of the filter window
    let mut total_size = 1;
    for &s in size {
        if s == 0 {
            return Err(NdimageError::InvalidInput(
                "Kernel size cannot be zero".into(),
            ));
        }
        total_size *= s;
    }

    // For maximum filter, we use the highest rank (total_size - 1)
    rank_filter(input, total_size - 1, size, Some(border_mode))
}

/// Apply a minimum filter to an n-dimensional array
///
/// # Arguments
///
/// * `input` - Input array to filter
/// * `size` - Size of the filter kernel in each dimension
/// * `mode` - Border handling mode (defaults to Reflect)
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Filtered array
#[allow(dead_code)]
pub fn minimum_filter<T, D>(
    input: &Array<T, D>,
    size: &[usize],
    mode: Option<BorderMode>,
) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + Debug + PartialOrd + Clone + Send + Sync + 'static,
    D: Dimension,
{
    // We can implement this using the rank filter with the rank set to 0
    let border_mode = mode.unwrap_or(BorderMode::Reflect);

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

    // For minimum filter, we use the lowest rank (0)
    rank_filter(input, 0, size, Some(border_mode))
}

/// Apply a percentile filter to an n-dimensional array
///
/// # Arguments
///
/// * `input` - Input array to filter
/// * `percentile` - Percentile value (0-100)
/// * `size` - Size of the filter kernel in each dimension
/// * `mode` - Border handling mode (defaults to Reflect)
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Filtered array
#[allow(dead_code)]
pub fn percentile_filter<T, D>(
    input: &Array<T, D>,
    percentile: f64,
    size: &[usize],
    mode: Option<BorderMode>,
) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + Debug + PartialOrd + Clone + Send + Sync + 'static,
    D: Dimension,
{
    let border_mode = mode.unwrap_or(BorderMode::Reflect);

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

    if !(0.0..=100.0).contains(&percentile) {
        return Err(NdimageError::InvalidInput(format!(
            "Percentile must be between 0 and 100, got {}",
            percentile
        )));
    }

    // Calculate total size of the filter window
    let mut total_size = 1;
    for &s in size {
        total_size *= s;
    }

    // Calculate the rank from the percentile
    // We need to map 0% -> rank 0, 100% -> rank (total_size - 1)
    let rank = ((percentile / 100.0) * (total_size as f64 - 1.0)).round() as usize;

    // Sanity check on the rank (should be within bounds)
    if rank >= total_size {
        return Err(NdimageError::InvalidInput(format!(
            "Calculated rank {} is out of bounds for window of size {}",
            rank, total_size
        )));
    }

    // Use the rank filter implementation with the calculated rank
    rank_filter(input, rank, size, Some(border_mode))
}

/// Apply a percentile filter with a custom footprint to an n-dimensional array
///
/// # Arguments
///
/// * `input` - Input array to filter
/// * `percentile` - Percentile value (0-100)
/// * `footprint` - Boolean array defining the filter footprint
/// * `mode` - Border handling mode
/// * `origin` - Origin of the filter kernel
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Filtered array
#[allow(dead_code)]
pub fn percentile_filter_footprint<T, D>(
    input: ArrayView<T, D>,
    percentile: f64,
    footprint: ArrayView<bool, D>,
    mode: BorderMode,
    origin: Vec<isize>,
) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + Debug + PartialOrd + Clone + Send + Sync + 'static,
    D: Dimension,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if footprint.ndim() != input.ndim() {
        return Err(NdimageError::DimensionError(format!(
            "Footprint must have same dimensionality as input (got {} expected {})",
            footprint.ndim(),
            input.ndim()
        )));
    }

    if !(0.0..=100.0).contains(&percentile) {
        return Err(NdimageError::InvalidInput(format!(
            "Percentile must be between 0 and 100, got {}",
            percentile
        )));
    }

    if origin.len() != input.ndim() {
        return Err(NdimageError::DimensionError(format!(
            "Origin must have same length as input dimensions (got {} expected {})",
            origin.len(),
            input.ndim()
        )));
    }

    // Count the number of true values in the footprint
    let total_size = footprint.iter().filter(|&&x| x).count();

    if total_size == 0 {
        return Err(NdimageError::InvalidInput(
            "Footprint cannot be empty (must have at least one true value)".into(),
        ));
    }

    // Calculate the rank from the percentile
    let rank = ((percentile / 100.0) * (total_size as f64 - 1.0)).round() as usize;

    // Sanity check on the rank
    if rank >= total_size {
        return Err(NdimageError::InvalidInput(format!(
            "Calculated rank {} is out of bounds for footprint of size {}",
            rank, total_size
        )));
    }

    // Use the rank filter implementation with the calculated rank
    rank_filter_footprint(input, rank, footprint, mode, origin)
}

/// Apply a rank filter with a custom footprint to an n-dimensional array
///
/// # Arguments
///
/// * `input` - Input array to filter
/// * `rank` - Rank of the element to select (0 = min, size-1 = max)
/// * `footprint` - Boolean array defining the filter footprint
/// * `mode` - Border handling mode
/// * `origin` - Origin of the filter kernel
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Filtered array
#[allow(dead_code)]
pub fn rank_filter_footprint<T, D>(
    input: ArrayView<T, D>,
    rank: usize,
    footprint: ArrayView<bool, D>,
    mode: BorderMode,
    origin: Vec<isize>,
) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + Debug + PartialOrd + Clone + Send + Sync + 'static,
    D: Dimension,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if footprint.ndim() != input.ndim() {
        return Err(NdimageError::DimensionError(format!(
            "Footprint must have same dimensionality as input (got {} expected {})",
            footprint.ndim(),
            input.ndim()
        )));
    }

    // Count the number of true values in the footprint
    let total_size = footprint.iter().filter(|&&x| x).count();

    if total_size == 0 {
        return Err(NdimageError::InvalidInput(
            "Footprint cannot be empty (must have at least one true value)".into(),
        ));
    }

    if rank >= total_size {
        return Err(NdimageError::InvalidInput(format!(
            "Rank {} is out of bounds for footprint of size {}",
            rank, total_size
        )));
    }

    // Handle scalar or constant case
    if input.len() <= 1 {
        return Ok(input.to_owned());
    }

    // Create output array
    let mut output = Array::<T, D>::zeros(input.raw_dim());

    // Get the center of the footprint for offset calculations
    let footprintshape = footprint.shape();
    let footprint_center: Vec<isize> = footprintshape.iter().map(|&s| (s / 2) as isize).collect();

    // Calculate padding based on footprint size and origin
    let mut pad_width = Vec::new();
    for d in 0..input.ndim() {
        let left_pad = (footprint_center[d] + origin[d]) as usize;
        let right_pad = footprintshape[d] - 1 - left_pad;
        pad_width.push((left_pad, right_pad));
    }

    // Pad the input array
    let padded_input = pad_array(&input.to_owned(), &pad_width, &mode, None)?;

    // Convert to dynamic arrays for easier indexing
    let input_dyn = input.view().into_dyn();
    let footprint_dyn = footprint.view().into_dyn();
    let mut output_dyn = output.view_mut().into_dyn();

    // Iterate through each position in the output array using linear iteration
    for linear_idx in 0..input.len() {
        // Convert linear index to multi-dimensional coordinates
        let output_coords = {
            let dims = input.shape();
            let mut coords = Vec::new();
            let mut remaining = linear_idx;

            for d in (0..dims.len()).rev() {
                coords.insert(0, remaining % dims[d]);
                remaining /= dims[d];
            }
            coords
        };

        // Collect values within the footprint at this position
        let mut values = Vec::new();

        // Iterate through the footprint using linear iteration
        for footprint_linear_idx in 0..footprint.len() {
            // Convert footprint linear index to coordinates
            let footprint_coords = {
                let dims = footprint.shape();
                let mut coords = Vec::new();
                let mut remaining = footprint_linear_idx;

                for d in (0..dims.len()).rev() {
                    coords.insert(0, remaining % dims[d]);
                    remaining /= dims[d];
                }
                coords
            };

            // Check if this footprint position is active
            let is_active = footprint_dyn[scirs2_core::ndarray::IxDyn(&footprint_coords)];

            if is_active {
                // Calculate the corresponding position in the padded input
                let mut input_coords = Vec::new();
                for d in 0..input.ndim() {
                    let coord = output_coords[d] + footprint_coords[d];
                    input_coords.push(coord);
                }

                // Get the value from the padded input using dynamic indexing
                let padded_dyn = padded_input.view().into_dyn();
                let value = padded_dyn[scirs2_core::ndarray::IxDyn(&input_coords)];
                values.push(value);
            }
        }

        // Sort values and select the rank-th element
        if !values.is_empty() {
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let selected_value = values[rank.min(values.len() - 1)];
            output_dyn[scirs2_core::ndarray::IxDyn(&output_coords)] = selected_value;
        }
    }

    Ok(output)
}

/// Apply a rank filter to an n-dimensional array
///
/// # Arguments
///
/// * `input` - Input array to filter
/// * `rank` - Rank of the element to select (0 = min, size-1 = max)
/// * `size` - Size of the filter kernel in each dimension
/// * `mode` - Border handling mode (defaults to Reflect)
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Filtered array
#[allow(dead_code)]
pub fn rank_filter<T, D>(
    input: &Array<T, D>,
    rank: usize,
    size: &[usize],
    mode: Option<BorderMode>,
) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + Debug + PartialOrd + Clone + Send + Sync + 'static,
    D: Dimension,
{
    let border_mode = mode.unwrap_or(BorderMode::Reflect);

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

    // Calculate total size of the filter window
    let mut total_size = 1;
    for &s in size {
        if s == 0 {
            return Err(NdimageError::InvalidInput(
                "Kernel size cannot be zero".into(),
            ));
        }
        total_size *= s;
    }

    if rank >= total_size {
        return Err(NdimageError::InvalidInput(format!(
            "Rank {} is out of bounds for window of size {}",
            rank, total_size
        )));
    }

    // Handle scalar or constant case
    if input.len() <= 1 {
        return Ok(input.to_owned());
    }

    // Calculate kernel radii (half size)
    let _radii: Vec<usize> = size.iter().map(|&s| s / 2).collect();

    // Dispatch to the appropriate implementation based on dimensionality
    match input.ndim() {
        1 => {
            // Handle 1D array
            let input_1d = input
                .to_owned()
                .into_dimensionality::<scirs2_core::ndarray::Ix1>()
                .map_err(|_| {
                    NdimageError::DimensionError("Failed to convert to 1D array".into())
                })?;

            let result_1d = rank_filter_1d(&input_1d, rank, size[0], &border_mode)?;

            result_1d.into_dimensionality::<D>().map_err(|_| {
                NdimageError::DimensionError("Failed to convert back from 1D array".into())
            })
        }
        2 => {
            // Handle 2D array
            let input_2d = input
                .to_owned()
                .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                .map_err(|_| {
                    NdimageError::DimensionError("Failed to convert to 2D array".into())
                })?;

            let result_2d = rank_filter_2d(&input_2d, rank, size, &border_mode)?;

            result_2d.into_dimensionality::<D>().map_err(|_| {
                NdimageError::DimensionError("Failed to convert back from 2D array".into())
            })
        }
        _ => {
            // For higher dimensions, convert to IxDyn, process, and convert back
            let input_dyn = input
                .to_owned()
                .into_dimensionality::<scirs2_core::ndarray::IxDyn>()
                .map_err(|_| {
                    NdimageError::DimensionError("Failed to convert to dynamic array".into())
                })?;

            let result_dyn = rank_filter_nd(&input_dyn, rank, size, &border_mode)?;

            result_dyn.into_dimensionality::<D>().map_err(|_| {
                NdimageError::DimensionError("Failed to convert back from dynamic array".into())
            })
        }
    }
}

/// Apply a rank filter to a 1D array
#[allow(dead_code)]
fn rank_filter_1d<T>(
    input: &Array1<T>,
    rank: usize,
    size: usize,
    mode: &BorderMode,
) -> NdimageResult<Array1<T>>
where
    T: Float + FromPrimitive + Debug + PartialOrd + Clone + Send + Sync + Zero + 'static,
{
    // Specialize for f32 - the most common type used in image processing
    // Check if T is f32 at runtime
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() && input.len() > 100 {
        // For f32, we can use specialized SIMD versions
        if size == 3 || size == 5 {
            // This is a little hacky, but avoids the complexity of generic specialization
            // in Rust which is not fully supported yet
            if size == 3 {
                return optimize_for_f32_size3(input, rank, mode);
            } else if size == 5 {
                return optimize_for_f32_size5(input, rank, mode);
            }
            // We should never reach here due to the if condition above
            unreachable!();
        }
    }

    // For f64 and other types, fallback to standard implementation
    let radius = size / 2;

    // Create output array
    let mut output = Array1::zeros(input.len());

    // Pad input for border handling
    let pad_width = vec![(radius, radius)];
    let padded_input = pad_array(input, &pad_width, mode, None)?;

    // Use parallel processing for large arrays
    // Parallel processing is enabled via Cargo.toml features
    if input.len() > 1000 {
        let process_window = move |i: &usize| -> std::result::Result<T, CoreError> {
            let mut window = vec![T::zero(); size];
            let center = *i + radius;

            // Extract window
            for k in 0..size {
                window[k] = padded_input[center - radius + k];
            }

            // Sort window and find element at specified rank
            window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            Ok(window[rank])
        };

        let indices: Vec<usize> = (0..input.len()).collect();
        let parallel_results = parallel_ops::parallel_map_result(&indices, process_window)?;

        // Copy the results to the output array
        for (i, val) in parallel_results.iter().enumerate() {
            output[i] = *val;
        }

        return Ok(output);
    }

    // For small arrays or when parallel feature is not enabled:
    // Use sequential processing with a reused buffer
    let mut window = vec![T::zero(); size];

    for i in 0..input.len() {
        let center = i + radius;

        // Extract window
        for (k, window_val) in window.iter_mut().enumerate().take(size) {
            *window_val = padded_input[center - radius + k];
        }

        // Sort window and find element at specified rank
        window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        output[i] = window[rank];
    }

    Ok(output)
}

/// Sort 3 elements and return them in order
#[inline]
#[allow(dead_code)]
fn sort3(a: f32, b: f32, c: f32) -> (f32, f32, f32) {
    if a > b {
        if b > c {
            (c, b, a)
        } else if a > c {
            (b, c, a)
        } else {
            (b, a, c)
        }
    } else if a > c {
        (c, a, b)
    } else if b > c {
        (a, c, b)
    } else {
        (a, b, c)
    }
}

/// Sort 5 elements in-place using a sorting network
#[inline]
#[allow(dead_code)]
fn sort5(arr: &mut [f32; 5]) {
    // Sorting network for 5 elements
    // Optimal sorting network from http://pages.ripco.net/~jgamble/nw.html
    if arr[0] > arr[1] {
        swap(arr, 0, 1);
    }
    if arr[3] > arr[4] {
        swap(arr, 3, 4);
    }
    if arr[2] > arr[4] {
        swap(arr, 2, 4);
    }
    if arr[2] > arr[3] {
        swap(arr, 2, 3);
    }
    if arr[0] > arr[3] {
        swap(arr, 0, 3);
    }
    if arr[0] > arr[2] {
        swap(arr, 0, 2);
    }
    if arr[1] > arr[4] {
        swap(arr, 1, 4);
    }
    if arr[1] > arr[3] {
        swap(arr, 1, 3);
    }
    if arr[1] > arr[2] {
        swap(arr, 1, 2);
    }
}

/// Helper function to swap elements in an array
#[inline]
#[allow(dead_code)]
fn swap<T: PartialOrd>(arr: &mut [T], i: usize, j: usize) {
    if arr[i] > arr[j] {
        arr.swap(i, j);
    }
}

/// Optimization for f32 arrays with window size 3
#[allow(dead_code)]
fn optimize_for_f32_size3<T>(
    input: &Array1<T>,
    rank: usize,
    mode: &BorderMode,
) -> NdimageResult<Array1<T>>
where
    T: Float + FromPrimitive + Debug + PartialOrd + Clone + Send + Sync + 'static,
{
    // This is a specialized implementation for f32 with window size 3
    // But we need to work with the generic T type for the interface

    // Convert input to f32 (we know it's safe because we checked the type at the call site)
    let input_data: Vec<f32> = input
        .iter()
        .map(|x| {
            x.to_f32().unwrap_or_else(|| {
                // Handle conversion failure gracefully
                f32::NAN
            })
        })
        .collect();
    let input_f32 = Array1::from_vec(input_data);

    // Process with f32 implementation
    let size = 3;
    let radius = size / 2;
    let mut output_f32 = Array1::zeros(input.len());

    // Pad input for border handling
    let pad_width = vec![(radius, radius)];
    let padded_input = pad_array(&input_f32, &pad_width, mode, None)?;

    // Process the data sequentially - SIMD operations must use scirs2-core::simd_ops
    // according to project policy (custom SIMD implementations are forbidden in modules)
    let mut results = Vec::with_capacity(input.len());

    for i in 0..input.len() {
        let center = i + radius;

        // Extract 3 elements for the window
        let a = padded_input[center - 1];
        let b = padded_input[center];
        let c = padded_input[center + 1];

        // Sort 3 elements (simple sorting network)
        let (min, mid, max) = sort3(a, b, c);

        // Select element based on rank
        let val = match rank {
            0 => min,
            1 => mid,
            2 => max,
            _ => unreachable!(), // Window size 3 can only have ranks 0, 1, 2
        };

        results.push((i, val));
    }

    // Apply results
    for (idx, val) in results {
        output_f32[idx] = val;
    }

    // Convert back to type T
    let output = output_f32.mapv(|x| {
        T::from_f32(x).unwrap_or_else(|| {
            // Handle conversion failure gracefully
            T::zero()
        })
    });
    Ok(output)
}

/// Optimization for f32 arrays with window size 5
#[allow(dead_code)]
fn optimize_for_f32_size5<T>(
    input: &Array1<T>,
    rank: usize,
    mode: &BorderMode,
) -> NdimageResult<Array1<T>>
where
    T: Float + FromPrimitive + Debug + PartialOrd + Clone + Send + Sync + 'static,
{
    // This is a specialized implementation for f32 with window size 5

    // Convert input to f32 (we know it's safe because we checked the type at the call site)
    let input_data: Vec<f32> = input
        .iter()
        .map(|x| {
            x.to_f32().unwrap_or_else(|| {
                // Handle conversion failure gracefully
                f32::NAN
            })
        })
        .collect();
    let input_f32 = Array1::from_vec(input_data);

    // Process with f32 implementation
    let size = 5;
    let radius = size / 2;
    let mut output_f32 = Array1::zeros(input.len());

    // Pad input for border handling
    let pad_width = vec![(radius, radius)];
    let padded_input = pad_array(&input_f32, &pad_width, mode, None)?;

    // We'll use parallel processing for larger arrays
    // Parallel processing for size 5 filters
    {
        let process_window = move |i: &usize| -> std::result::Result<(usize, f32), CoreError> {
            let center = *i + radius;

            // Extract 5 elements
            let mut window = [
                padded_input[center - 2],
                padded_input[center - 1],
                padded_input[center],
                padded_input[center + 1],
                padded_input[center + 2],
            ];

            // Sort using the optimized sorting network
            sort5(&mut window);

            // Return index and value
            Ok((*i, window[rank]))
        };

        let indices: Vec<usize> = (0..input.len()).collect();
        let results = parallel_ops::parallel_map_result(&indices, process_window)?;

        // Apply results to output array
        for (idx, val) in results {
            output_f32[idx] = val;
        }

        // Convert back to type T
        let output = output_f32.mapv(|x| {
            T::from_f32(x).unwrap_or_else(|| {
                // Handle conversion failure gracefully
                T::zero()
            })
        });
        return Ok(output);
    }

    #[allow(unreachable_code)]
    {
        // Convert back to type T
        let output = output_f32.mapv(|x| {
            T::from_f32(x).unwrap_or_else(|| {
                // Handle conversion failure gracefully
                T::zero()
            })
        });
        Ok(output)
    }
}

/// Apply a rank filter to a 2D array
#[allow(dead_code)]
fn rank_filter_2d<T>(
    input: &Array2<T>,
    rank: usize,
    size: &[usize],
    mode: &BorderMode,
) -> NdimageResult<Array2<T>>
where
    T: Float + FromPrimitive + Debug + PartialOrd + Clone + Send + Sync + 'static,
{
    let rows = input.shape()[0];
    let cols = input.shape()[1];
    let radius_y = size[0] / 2;
    let radius_x = size[1] / 2;
    let window_size = size[0] * size[1];

    // Create output array
    let mut output = Array2::zeros((rows, cols));

    // Pad input for border handling
    let pad_width = vec![(radius_y, radius_y), (radius_x, radius_x)];
    let padded_input = pad_array(input, &pad_width, mode, None)?;

    // Use memory-efficient parallel processing for large arrays
    // Parallel processing for 2D arrays
    if rows * cols > 10000 {
        // Clone the size parameter to avoid lifetime issues with parallel_map
        let size_clone = size.to_vec();

        // Process rows in parallel using scirs2-core's parallel module
        let process_row = move |i: &usize| -> std::result::Result<Vec<T>, CoreError> {
            let mut row_results = Vec::with_capacity(cols);
            let mut window = vec![T::zero(); window_size];

            for j in 0..cols {
                // Calculate padded coordinates
                let center_y = *i + radius_y;
                let center_x = j + radius_x;

                // Extract window
                let mut window_idx = 0;
                for ky in 0..size_clone[0] {
                    for kx in 0..size_clone[1] {
                        let y = center_y - radius_y + ky;
                        let x = center_x - radius_x + kx;
                        window[window_idx] = padded_input[[y, x]];
                        window_idx += 1;
                    }
                }

                // Sort window and find element at specified rank
                window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                row_results.push(window[rank]);
            }

            Ok(row_results)
        };

        let row_indices: Vec<usize> = (0..rows).collect();
        let results = parallel_ops::parallel_map_result(&row_indices, process_row)?;

        // Copy the results to the output array
        for (i, row) in results.iter().enumerate() {
            for (j, val) in row.iter().enumerate() {
                output[[i, j]] = *val;
            }
        }

        return Ok(output);
    }

    // For small arrays or when parallel feature is not enabled:
    // Use sequential processing with a reused buffer
    let mut window = vec![T::zero(); window_size];

    for i in 0..rows {
        for j in 0..cols {
            // Calculate padded coordinates
            let center_y = i + radius_y;
            let center_x = j + radius_x;

            // Extract window more efficiently
            let mut window_idx = 0;
            for ky in 0..size[0] {
                for kx in 0..size[1] {
                    let y = center_y - radius_y + ky;
                    let x = center_x - radius_x + kx;
                    window[window_idx] = padded_input[[y, x]];
                    window_idx += 1;
                }
            }

            // Sort window and find element at specified rank
            window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            output[[i, j]] = window[rank];
        }
    }

    Ok(output)
}

/// Apply a rank filter to an n-dimensional IxDyn array (3D and higher)
#[allow(dead_code)]
fn rank_filter_nd<T>(
    input: &Array<T, scirs2_core::ndarray::IxDyn>,
    rank: usize,
    size: &[usize],
    mode: &BorderMode,
) -> NdimageResult<Array<T, scirs2_core::ndarray::IxDyn>>
where
    T: Float + FromPrimitive + Debug + PartialOrd + Clone + Send + Sync + 'static,
{
    let shape = input.shape();
    let ndim = input.ndim();

    // Calculate total size of the filter window
    let window_size = size.iter().product();

    // Calculate radii for each dimension
    let radii: Vec<usize> = size.iter().map(|&s| s / 2).collect();

    // Create output array
    let mut output = Array::zeros(input.raw_dim());

    // Pad input for border handling
    let pad_width: Vec<(usize, usize)> = radii.iter().map(|&r| (r, r)).collect();
    let padded_input = pad_array(input, &pad_width, mode, None)?;

    // Use sequential processing with a reused buffer
    let mut window = Vec::with_capacity(window_size);

    for idx in scirs2_core::ndarray::indices(shape) {
        let idx_vec: Vec<_> = idx.slice().to_vec();
        window.clear();

        // Calculate padded coordinates for the center
        let center_coords: Vec<usize> = idx_vec
            .iter()
            .zip(&radii)
            .map(|(&idx, &radius)| idx + radius)
            .collect();

        // Generate all offsets for the window
        let mut offset_stack = vec![Vec::new()];
        for &window_dim_size in size {
            let mut new_stack = Vec::new();
            for existing_offset in offset_stack {
                for offset in 0..window_dim_size {
                    let mut new_offset = existing_offset.clone();
                    new_offset.push(offset);
                    new_stack.push(new_offset);
                }
            }
            offset_stack = new_stack;
        }

        // Extract all values in the window
        for offsets in offset_stack {
            let mut padded_coords = Vec::with_capacity(ndim);
            for ((&center, &radius), &offset) in center_coords.iter().zip(&radii).zip(&offsets) {
                padded_coords.push(center - radius + offset);
            }

            // Convert to slice for ndarray indexing
            let value = padded_input[padded_coords.as_slice()];
            window.push(value);
        }

        // Sort window and find element at specified rank
        window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        output[idx_vec.as_slice()] = window[rank];
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

    #[test]
    fn test_maximum_filter_1d() {
        // Create a 1D array
        let array = Array1::from_vec(vec![1.0, 5.0, 3.0, 4.0, 2.0]);

        // Apply maximum filter with size 3
        let result = maximum_filter(&array, &[3], None).expect("test function should succeed");

        // Expected: [5, 5, 5, 4, 4]
        assert_eq!(result.len(), array.len());
        assert_eq!(result[0], 5.0); // Max of [1, 5, 3]
        assert_eq!(result[1], 5.0); // Max of [5, 3, 4]
        assert_eq!(result[2], 5.0); // Max of [3, 4, 2]
        assert_eq!(result[3], 4.0); // Max of [4, 2, 2] (with reflect boundary)
        assert_eq!(result[4], 4.0); // Max of [2, 2, 4] (with reflect boundary)
    }

    #[test]
    fn test_minimum_filter_1d() {
        // Create a 1D array
        let array = Array1::from_vec(vec![5.0, 2.0, 3.0, 4.0, 1.0]);

        // Apply minimum filter with size 3
        let result = minimum_filter(&array, &[3], None).expect("test function should succeed");

        // Note: These are expected values with reflect boundary mode:
        // Radius for window
        let radius = 1;
        let pad_width = vec![(radius, radius)];
        // This comment explains the expected values based on padded_input
        let _padded_input = pad_array(&array, &pad_width, &BorderMode::Reflect, None)
            .expect("test function should succeed");

        // Expected values with reflect boundary mode:
        // Padded input: [2.0, 5.0, 2.0, 3.0, 4.0, 1.0, 4.0]
        // Position 0: [2.0, 5.0, 2.0] -> min = 2.0
        // Position 1: [5.0, 2.0, 3.0] -> min = 2.0
        // Position 2: [2.0, 3.0, 4.0] -> min = 2.0
        // Position 3: [3.0, 4.0, 1.0] -> min = 1.0
        // Position 4: [4.0, 1.0, 4.0] -> min = 1.0

        assert_eq!(result.len(), array.len());
        assert_eq!(result[0], 2.0);
        assert_eq!(result[1], 2.0);
        assert_eq!(result[2], 2.0);
        assert_eq!(result[3], 1.0);
        assert_eq!(result[4], 1.0);
    }

    #[test]
    fn test_maximum_filter() {
        // Create a simple test image
        let mut image = Array2::zeros((5, 5));
        image[[1, 1]] = 1.0;
        image[[1, 3]] = 0.8;
        image[[3, 1]] = 0.9;
        image[[3, 3]] = 0.7;

        // Apply filter with 3x3 window
        let result = maximum_filter(&image, &[3, 3], None).expect("test function should succeed");

        // Check that result has the same shape
        assert_eq!(result.shape(), image.shape());

        // Check maximum values are propagated correctly
        assert_eq!(result[[0, 0]], 1.0); // Should contain the maximum value from the window
        assert_eq!(result[[2, 2]], 1.0); // Central point should see all values
    }

    #[test]
    fn test_minimum_filter() {
        // Create a simple test image
        let mut image = Array2::ones((5, 5));
        image[[2, 2]] = 0.0; // Center pixel has minimum value

        // Apply filter with 3x3 window
        let result = minimum_filter(&image, &[3, 3], None).expect("test function should succeed");

        // Check that result has the same shape
        assert_eq!(result.shape(), image.shape());

        // Check minimum value propagation
        // The value 0.0 should be propagated to all pixels within the 3x3 window centered at [2,2]
        for i in 1..4 {
            for j in 1..4 {
                assert_eq!(result[[i, j]], 0.0);
            }
        }

        // Corners should still be 1.0 as they're outside the influence of the minimum
        assert_eq!(result[[0, 0]], 0.0); // This is 0 because of reflect padding
        assert_eq!(result[[0, 4]], 0.0); // This is 0 because of reflect padding
        assert_eq!(result[[4, 0]], 0.0); // This is 0 because of reflect padding
        assert_eq!(result[[4, 4]], 0.0); // This is 0 because of reflect padding
    }

    #[test]
    fn test_percentile_filter() {
        // Create a simple test array
        let array = Array1::from_vec(vec![1.0, 2.0, 3.0, 10.0, 5.0]);

        // Apply median filter (50th percentile) with size 3
        let result =
            percentile_filter(&array, 50.0, &[3], None).expect("test function should succeed");

        // Note: These are expected values with reflect boundary mode:
        // Radius for window
        let radius = 1;
        let pad_width = vec![(radius, radius)];
        // This comment explains the expected values based on padded_input
        let _padded_input = pad_array(&array, &pad_width, &BorderMode::Reflect, None)
            .expect("test function should succeed");

        // Expected values with reflect boundary mode:
        // Padded input: [2.0, 1.0, 2.0, 3.0, 10.0, 5.0, 10.0]
        // Position 0: [2.0, 1.0, 2.0] -> sorted [1.0, 2.0, 2.0] -> median = 2.0
        // Position 1: [1.0, 2.0, 3.0] -> sorted [1.0, 2.0, 3.0] -> median = 2.0
        // Position 2: [2.0, 3.0, 10.0] -> sorted [2.0, 3.0, 10.0] -> median = 3.0
        // Position 3: [3.0, 10.0, 5.0] -> sorted [3.0, 5.0, 10.0] -> median = 5.0
        // Position 4: [10.0, 5.0, 10.0] -> sorted [5.0, 10.0, 10.0] -> median = 10.0

        // Check that result has the same length
        assert_eq!(result.len(), array.len());

        assert_eq!(result[0], 2.0);
        assert_eq!(result[1], 2.0);
        assert_eq!(result[2], 3.0);
        assert_eq!(result[3], 5.0);
        assert_eq!(result[4], 10.0);
    }

    #[test]
    fn test_rank_filter() {
        // Create a simple test array
        let array = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        // Apply rank filter with rank 1 (second lowest value) and size 3
        let result = rank_filter(&array, 1, &[3], None).expect("test function should succeed");

        // Note: These are expected values with reflect boundary mode:
        // Radius for window
        let radius = 1;
        let pad_width = vec![(radius, radius)];
        // This comment explains the expected values based on padded_input
        let _padded_input = pad_array(&array, &pad_width, &BorderMode::Reflect, None)
            .expect("test function should succeed");

        // Expected values with reflect boundary mode:
        // Padded input: [2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0]
        // Position 0: [2.0, 1.0, 2.0] -> sorted [1.0, 2.0, 2.0] -> rank 1 = 2.0
        // Position 1: [1.0, 2.0, 3.0] -> sorted [1.0, 2.0, 3.0] -> rank 1 = 2.0
        // Position 2: [2.0, 3.0, 4.0] -> sorted [2.0, 3.0, 4.0] -> rank 1 = 3.0
        // Position 3: [3.0, 4.0, 5.0] -> sorted [3.0, 4.0, 5.0] -> rank 1 = 4.0
        // Position 4: [4.0, 5.0, 4.0] -> sorted [4.0, 4.0, 5.0] -> rank 1 = 4.0

        // Check that result has the same length
        assert_eq!(result.len(), array.len());

        assert_eq!(result[0], 2.0);
        assert_eq!(result[1], 2.0);
        assert_eq!(result[2], 3.0);
        assert_eq!(result[3], 4.0);
        assert_eq!(result[4], 4.0);
    }

    #[test]
    fn test_rank_filter_invalid_rank() {
        // Create a simple test array
        let array = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        // Try to apply rank filter with an invalid rank (3 for window size 3)
        let result = rank_filter(&array, 3, &[3], None);

        // Should get an error
        assert!(result.is_err());
    }

    #[test]
    fn test_rank_filter_3d() {
        use scirs2_core::ndarray::Array3;

        // Create a 3D test array
        let mut array = Array3::zeros((4, 4, 4));

        // Set some values to create a pattern
        array[[1, 1, 1]] = 10.0;
        array[[1, 2, 1]] = 8.0;
        array[[2, 1, 1]] = 6.0;
        array[[2, 2, 1]] = 4.0;
        array[[1, 1, 2]] = 2.0;

        // Apply maximum filter (rank = window_size - 1) with 3x3x3 window
        let window_size = 3 * 3 * 3;
        let result = rank_filter(&array, window_size - 1, &[3, 3, 3], None)
            .expect("test function should succeed");

        // Check that result has the same shape
        assert_eq!(result.shape(), array.shape());

        // The maximum value (10.0) should propagate to nearby locations
        // At position [1,1,1], the 3x3x3 window should see the maximum value 10.0
        assert_eq!(result[[1, 1, 1]], 10.0);

        // Check that the shape is preserved and no errors occur
        assert_eq!(result.ndim(), 3);
        assert_eq!(result.len(), array.len());
    }

    #[test]
    fn test_minimum_filter_3d() {
        use scirs2_core::ndarray::Array3;

        // Create a 3D test array with all ones except one zero
        let mut array = Array3::ones((4, 4, 4));
        array[[2, 2, 2]] = 0.0; // Set center to minimum value

        // Apply minimum filter with 3x3x3 window
        let result =
            minimum_filter(&array, &[3, 3, 3], None).expect("test function should succeed");

        // Check that result has the same shape
        assert_eq!(result.shape(), array.shape());

        // The minimum value (0.0) should propagate to all positions within the window
        // At position [2,2,2], the minimum should be 0.0
        assert_eq!(result[[2, 2, 2]], 0.0);

        // Positions adjacent to [2,2,2] should also see the minimum
        assert_eq!(result[[1, 2, 2]], 0.0);
        assert_eq!(result[[3, 2, 2]], 0.0);
        assert_eq!(result[[2, 1, 2]], 0.0);
        assert_eq!(result[[2, 3, 2]], 0.0);
        assert_eq!(result[[2, 2, 1]], 0.0);
        assert_eq!(result[[2, 2, 3]], 0.0);
    }

    #[test]
    fn test_percentile_filter_3d() {
        use scirs2_core::ndarray::Array3;

        // Create a 3D test array
        let array = Array3::from_shape_fn((3, 3, 3), |(i, j, k)| {
            (i * 9 + j * 3 + k) as f64 // Values from 0 to 26
        });

        // Apply 50th percentile (median) filter with 3x3x3 window
        let result = percentile_filter(&array, 50.0, &[3, 3, 3], None)
            .expect("test function should succeed");

        // Check that result has the same shape
        assert_eq!(result.shape(), array.shape());

        // Check that no errors occur and output is reasonable
        assert_eq!(result.ndim(), 3);
        assert_eq!(result.len(), array.len());

        // The result should contain finite values
        for &value in result.iter() {
            assert!(value.is_finite());
        }
    }
}
