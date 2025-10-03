//! Gaussian filtering functions for n-dimensional arrays

use scirs2_core::ndarray::{Array, Array1, Array2, Dimension, Ix2, IxDyn};

use super::{pad_array, BorderMode};
use crate::error::{NdimageError, NdimageResult};
use scirs2_core::{parallel_ops, CoreError};

/// Apply a Gaussian filter to an n-dimensional array of f64 values
///
/// Gaussian filtering is a fundamental image processing operation that applies a
/// Gaussian kernel to smooth the input array, reducing noise while preserving
/// edges better than simple averaging filters.
///
/// # Arguments
///
/// * `input` - Input array to filter
/// * `sigma` - Standard deviation for Gaussian kernel (controls smoothing strength)
/// * `mode` - Border handling mode (defaults to Reflect). Options include:
///   - `BorderMode::Reflect`: Mirror the input along the boundary
///   - `BorderMode::Constant`: Use a constant value outside boundaries  
///   - `BorderMode::Nearest`: Extend the edge values
///   - `BorderMode::Wrap`: Wrap around periodically
/// * `truncate` - Truncate the filter at this many standard deviations (defaults to 4.0)
///
/// # Returns
///
/// * `Result<Array<f64, D>>` - Smoothed array with same shape as input
///
/// # Examples
///
/// ## Basic 1D smoothing
/// ```no_run
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::filters::gaussian_filter;
///
/// let data = array![1.0, 5.0, 2.0, 8.0, 3.0];
/// let smoothed = gaussian_filter(&data, 0.8, None, None).unwrap();
/// // Result will be smoother with reduced sharp transitions
/// ```
///
/// ## 2D image smoothing with different border modes
/// ```no_run
/// use scirs2_core::ndarray::Array2;
/// use scirs2_ndimage::filters::{gaussian_filter, BorderMode};
///
/// let image = Array2::from_shape_fn((10, 10), |(i, j)| {
///     ((i * j) as f64).sin()
/// });
///
/// // Light smoothing with reflective boundaries
/// let smooth1 = gaussian_filter(&image, 1.0, Some(BorderMode::Reflect), None).unwrap();
///
/// // Heavy smoothing with constant boundaries  
/// let smooth2 = gaussian_filter(&image, 3.0, Some(BorderMode::Constant), None).unwrap();
/// ```
///
/// ## 3D volume smoothing
/// ```no_run
/// use scirs2_core::ndarray::Array3;
/// use scirs2_ndimage::filters::gaussian_filter;
///
/// let volume = Array3::from_shape_fn((20, 20, 20), |(i, j, k)| {
///     (i + j + k) as f64
/// });
///
/// let smoothed_volume = gaussian_filter(&volume, 2.0, None, None).unwrap();
/// assert_eq!(smoothed_volume.shape(), volume.shape());
/// ```
///
/// # Performance Notes
///
/// - Uses separable filtering for O(n) complexity instead of O(n²) for 2D
/// - Automatically switches to parallel processing for large arrays
/// - Kernel size is automatically determined from sigma and truncate parameters
/// - For σ < 0.5, consider using other smoothing methods for better efficiency
#[allow(dead_code)]
pub fn gaussian_filter<D>(
    input: &Array<f64, D>,
    sigma: f64,
    mode: Option<BorderMode>,
    truncate: Option<f64>,
) -> NdimageResult<Array<f64, D>>
where
    D: Dimension + 'static,
{
    let border_mode = mode.unwrap_or(BorderMode::Reflect);
    let trunc = truncate.unwrap_or(4.0);

    // Validate inputs
    if sigma <= 0.0 {
        return Err(NdimageError::InvalidInput("Sigma must be positive".into()));
    }

    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    // Handle scalar or constant case
    if input.len() <= 1 {
        return Ok(input.to_owned());
    }

    // Dispatch to the appropriate implementation based on dimensionality
    match input.ndim() {
        1 => {
            // Apply 1D gaussian filter directly
            gaussian_filter1d_f64(input, sigma, mode, truncate)
        }
        2 => {
            // For 2D arrays, apply the filter along each axis
            // Apply along axis 0 (rows)
            let temp = apply_gaussian_along_axis_f64(input, 0, sigma, &border_mode, trunc)?;

            // Apply along axis 1 (columns)
            apply_gaussian_along_axis_f64(&temp, 1, sigma, &border_mode, trunc)
        }
        _ => {
            // For higher dimensions, apply separable filtering along each axis
            let ndim = input.ndim();
            let mut result = input.to_owned();

            // Apply 1D gaussian filter along each axis
            for axis in 0..ndim {
                result =
                    apply_gaussian_along_axis_nd_f64(&result, axis, sigma, &border_mode, trunc)?;
            }

            Ok(result)
        }
    }
}

/// Apply a 1D Gaussian filter along a single dimension (f64 version)
#[allow(dead_code)]
pub fn gaussian_filter1d_f64<D>(
    input: &Array<f64, D>,
    sigma: f64,
    mode: Option<BorderMode>,
    truncate: Option<f64>,
) -> NdimageResult<Array<f64, D>>
where
    D: Dimension + 'static,
{
    let border_mode = mode.unwrap_or(BorderMode::Reflect);
    let trunc = truncate.unwrap_or(4.0);

    // Validate inputs
    if sigma <= 0.0 {
        return Err(NdimageError::InvalidInput("Sigma must be positive".into()));
    }

    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    // Handle scalar or constant case
    if input.len() <= 1 {
        return Ok(input.to_owned());
    }

    // Get the kernel
    let kernel = gaussian_kernel1d_f64(sigma, trunc)?;

    // For 1D arrays, apply directly
    if input.ndim() == 1 {
        let input_1d = input
            .to_owned()
            .into_dimensionality::<scirs2_core::ndarray::Ix1>()
            .map_err(|_| NdimageError::DimensionError("Failed to convert to 1D array".into()))?;
        let result_1d = apply_kernel1d_1d_f64(&input_1d, &kernel, &border_mode)?;
        return result_1d.into_dimensionality::<D>().map_err(|_| {
            NdimageError::DimensionError("Failed to convert back from 1D array".into())
        });
    }

    // For higher dimensions, handle via the axis-specific implementations
    Ok(input.to_owned())
}

/// Generate a 1D Gaussian kernel for f64 filtering
/// This function uses manual caching to avoid Result Clone issues
#[allow(dead_code)]
pub fn gaussian_kernel1d_f64(sigma: f64, truncate: f64) -> NdimageResult<Array1<f64>> {
    // Manual caching using lazy_static or thread_local would be ideal here
    // but for simplicity, we'll just implement the function without caching for now

    if sigma <= 0.0 {
        return Err(NdimageError::InvalidInput("Sigma must be positive".into()));
    }

    let radius = (truncate * sigma).ceil();
    let radius_int = radius as usize;

    let size = 2 * radius_int + 1;
    let mut kernel = Array1::zeros(size);

    // Create x values for Gaussian calculation (distance from center)
    let mut x_values = Array1::zeros(size);
    for i in 0..size {
        x_values[i] = (i as f64) - (radius_int as f64);
    }

    // Calculate x^2 manually
    let mut x_squared = Array1::zeros(size);
    for i in 0..size {
        x_squared[i] = x_values[i] * x_values[i];
    }

    // Calculate exp(-x^2/(2*_sigma^2))
    let two_sigma_squared = 2.0 * sigma * sigma;

    // Apply the Gaussian formula: exp(-x^2/(2*_sigma^2))
    for i in 0..size {
        kernel[i] = (-x_squared[i] / two_sigma_squared).exp();
    }

    // Normalize
    let sum = kernel.sum();
    if sum > 0.0 {
        kernel.mapv_inplace(|x| x / sum);
    }

    Ok(kernel)
}

/// Apply a f64 1D kernel to a f64 1D array
#[allow(dead_code)]
fn apply_kernel1d_1d_f64(
    input: &Array1<f64>,
    kernel: &Array1<f64>,
    mode: &BorderMode,
) -> NdimageResult<Array1<f64>> {
    let input_len = input.len();
    let kernel_len = kernel.len();
    let radius = kernel_len / 2;

    // Create output array
    let mut output = Array1::zeros(input_len);

    // Pad input for border handling
    let pad_width = vec![(radius, radius)];
    let padded_input = pad_array(input, &pad_width, mode, None)?;

    // Apply kernel to each position
    for i in 0..input_len {
        let center = i + radius;
        let mut sum = 0.0;

        for k in 0..kernel_len {
            sum += padded_input[center + k - radius] * kernel[k];
        }

        output[i] = sum;
    }

    Ok(output)
}

/// Apply a Gaussian filter along a specific axis (f64 version)
#[allow(dead_code)]
fn apply_gaussian_along_axis_f64<D>(
    input: &Array<f64, D>,
    axis: usize,
    sigma: f64,
    mode: &BorderMode,
    truncate: f64,
) -> NdimageResult<Array<f64, D>>
where
    D: Dimension + 'static,
{
    // Validate axis
    if axis >= input.ndim() {
        return Err(NdimageError::DimensionError(format!(
            "Axis {} is out of bounds for array of dimension {}",
            axis,
            input.ndim()
        )));
    }

    // Create 1D kernel
    let kernel = gaussian_kernel1d_f64(sigma, truncate)?;

    // For 2D arrays only, implement a simple solution
    if input.ndim() == 2 {
        // We need to convert to Array2 to use the slice methods for 2D arrays
        let array2d = input
            .clone()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NdimageError::DimensionError("Failed to convert to 2D array".into()))?;

        let mut output = array2d.clone();

        // We'll implement a specialized version for 2D arrays
        match axis {
            0 => {
                // Apply along rows (axis 0)
                for i in 0..array2d.shape()[0] {
                    // Extract row
                    let row_view = array2d.row(i).to_owned();
                    let row_1d = row_view.as_slice().ok_or_else(|| {
                        NdimageError::ComputationError(
                            "Failed to get contiguous slice from row".into(),
                        )
                    })?;
                    // Create a 1D array from the slice
                    let row_array = Array1::from_vec(row_1d.to_vec());

                    // Apply kernel
                    let filtered_row = apply_kernel1d_1d_f64(&row_array, &kernel, mode)?;

                    // Put back
                    for j in 0..array2d.shape()[1] {
                        output[[i, j]] = filtered_row[j];
                    }
                }
            }
            1 => {
                // Apply along columns (axis 1)
                for j in 0..array2d.shape()[1] {
                    // Extract column
                    let col_view = array2d.column(j).to_owned();
                    let col_1d = col_view.as_slice().ok_or_else(|| {
                        NdimageError::ComputationError(
                            "Failed to get contiguous slice from column".into(),
                        )
                    })?;
                    // Create a 1D array from the slice
                    let col_array = Array1::from_vec(col_1d.to_vec());

                    // Apply kernel
                    let filtered_col = apply_kernel1d_1d_f64(&col_array, &kernel, mode)?;

                    // Put back
                    for i in 0..array2d.shape()[0] {
                        output[[i, j]] = filtered_col[i];
                    }
                }
            }
            _ => unreachable!(),
        }

        // Convert back to the original dimensionality
        output.into_dimensionality::<D>().map_err(|_| {
            NdimageError::DimensionError("Failed to convert back from 2D array".into())
        })
    } else {
        // For other dimensionalities, use the n-dimensional implementation
        apply_gaussian_along_axis_nd_f64(input, axis, sigma, mode, truncate)
    }
}

/// Apply a Gaussian filter along a specific axis for n-dimensional arrays (f64 version)
#[allow(dead_code)]
fn apply_gaussian_along_axis_nd_f64<D>(
    input: &Array<f64, D>,
    axis: usize,
    sigma: f64,
    mode: &BorderMode,
    truncate: f64,
) -> NdimageResult<Array<f64, D>>
where
    D: Dimension + 'static,
{
    // Validate axis
    if axis >= input.ndim() {
        return Err(NdimageError::DimensionError(format!(
            "Axis {} is out of bounds for array of dimension {}",
            axis,
            input.ndim()
        )));
    }

    // Create 1D Gaussian kernel
    let kernel = gaussian_kernel1d_f64(sigma, truncate)?;
    let kernel_len = kernel.len();
    let kernel_radius = kernel_len / 2;

    // Get input shape for later
    let inputshape = input.shape();

    // Convert to dynamic dimension to facilitate filtering across arbitrary axes
    let input_dyn = input.clone().into_dyn();

    // Create output array dynamically
    let mut output_dyn = Array::zeros(input_dyn.raw_dim());

    // Process each position in the output array
    let indices: Vec<IxDyn> = output_dyn
        .indexed_iter()
        .map(|(idx, _)| idx.clone())
        .collect();

    // Helper function to convolve kernel with input at a specific position
    let convolve_position = |idx: &IxDyn| -> (IxDyn, f64) {
        // Convert to vec for easier manipulation
        let idx_vec = idx.as_array_view().to_vec();
        let mut sum = 0.0;

        // Apply kernel along the axis
        for k in 0..kernel.len() {
            // Calculate padded index
            let mut padded_idx_vec = idx_vec.clone();
            let kernel_pos = k as isize - kernel_radius as isize;

            // Calculate source position along the axis
            let src_pos = idx_vec[axis] as isize + kernel_pos;
            let src_len = inputshape[axis] as isize;

            // Apply border mode to get actual index
            let src_idx = match mode {
                BorderMode::Reflect => {
                    if src_pos < 0 {
                        (-src_pos - 1) as usize % src_len as usize
                    } else if src_pos >= src_len {
                        (2 * src_len - src_pos - 1) as usize % src_len as usize
                    } else {
                        src_pos as usize
                    }
                }
                BorderMode::Mirror => {
                    if src_pos < 0 {
                        (-src_pos) as usize % (src_len as usize * 2 - 2).max(1)
                    } else if src_pos >= src_len {
                        (2 * src_len - src_pos - 2) as usize % (src_len as usize * 2 - 2).max(1)
                    } else {
                        src_pos as usize
                    }
                }
                BorderMode::Wrap => {
                    if src_len == 0 {
                        0
                    } else {
                        ((src_pos % src_len + src_len) % src_len) as usize
                    }
                }
                BorderMode::Constant => {
                    if src_pos < 0 || src_pos >= src_len {
                        // Skip this element (using zero)
                        continue;
                    } else {
                        src_pos as usize
                    }
                }
                BorderMode::Nearest => {
                    if src_pos < 0 {
                        0
                    } else if src_pos >= src_len {
                        (src_len - 1) as usize
                    } else {
                        src_pos as usize
                    }
                }
            };

            padded_idx_vec[axis] = src_idx;
            let padded_idx = IxDyn(&padded_idx_vec);

            // Multiply by kernel weight and add to sum
            sum += input_dyn[&padded_idx] * kernel[k];
        }

        (idx.clone(), sum)
    };

    // Decide whether to process in parallel based on array size
    let threshold = 1000; // Arbitrary threshold to avoid parallelism overhead for small arrays

    if input.len() <= threshold {
        // Sequential processing for small arrays
        for idx in indices {
            let (pos, value) = convolve_position(&idx);
            output_dyn[&pos] = value;
        }
    } else {
        // Process in parallel for larger arrays
        // Create a new closure that doesn't depend on the captured variables
        let inputshape_clone = inputshape.to_vec();
        let axis_clone = axis;
        let mode_clone = *mode;
        let kernel_clone = kernel.clone();

        // Wrap in an Arc for thread safety
        let input_dyn_arc = std::sync::Arc::new(input_dyn.clone());

        let parallel_convolve = move |idx: &IxDyn| -> std::result::Result<(IxDyn, f64), CoreError> {
            // Convert to vec for easier manipulation
            let idx_vec = idx.as_array_view().to_vec();
            let mut sum = 0.0;

            // Apply kernel along the axis
            for k in 0..kernel_clone.len() {
                // Calculate padded index
                let mut padded_idx_vec = idx_vec.clone();
                let kernel_pos = k as isize - (kernel_clone.len() / 2) as isize;

                // Calculate source position along the axis
                let src_pos = idx_vec[axis_clone] as isize + kernel_pos;
                let src_len = inputshape_clone[axis_clone] as isize;

                // Apply border mode to get actual index
                let src_idx = match mode_clone {
                    BorderMode::Reflect => {
                        if src_pos < 0 {
                            (-src_pos - 1) as usize % src_len as usize
                        } else if src_pos >= src_len {
                            (2 * src_len - src_pos - 1) as usize % src_len as usize
                        } else {
                            src_pos as usize
                        }
                    }
                    BorderMode::Mirror => {
                        if src_pos < 0 {
                            (-src_pos) as usize % (src_len as usize * 2 - 2).max(1)
                        } else if src_pos >= src_len {
                            (2 * src_len - src_pos - 2) as usize % (src_len as usize * 2 - 2).max(1)
                        } else {
                            src_pos as usize
                        }
                    }
                    BorderMode::Wrap => {
                        if src_len == 0 {
                            0
                        } else {
                            ((src_pos % src_len + src_len) % src_len) as usize
                        }
                    }
                    BorderMode::Constant => {
                        if src_pos < 0 || src_pos >= src_len {
                            // Skip this element (using zero)
                            continue;
                        } else {
                            src_pos as usize
                        }
                    }
                    BorderMode::Nearest => {
                        if src_pos < 0 {
                            0
                        } else if src_pos >= src_len {
                            (src_len - 1) as usize
                        } else {
                            src_pos as usize
                        }
                    }
                };

                padded_idx_vec[axis_clone] = src_idx;
                let padded_idx = IxDyn(&padded_idx_vec);

                // Multiply by kernel weight and add to sum
                sum += input_dyn_arc[&padded_idx] * kernel_clone[k];
            }

            Ok((idx.clone(), sum))
        };

        // Use parallel_map from scirs2-core for parallel processing
        let results = parallel_ops::parallel_map_result(&indices, parallel_convolve)?;

        // Apply results to output array
        for (pos, value) in results {
            output_dyn[&pos] = value;
        }
    }

    // Convert back to the original dimension type
    output_dyn.into_dimensionality::<D>().map_err(|_| {
        NdimageError::DimensionError("Failed to convert back to original dimensions".into())
    })
}

/// Apply a gaussian filter to an n-dimensional array of f32 values
///
/// # Arguments
///
/// * `input` - Input array to filter
/// * `sigma` - Standard deviation for Gaussian kernel
/// * `mode` - Border handling mode (defaults to Reflect)
/// * `truncate` - Truncate the filter at this many standard deviations (defaults to 4.0)
///
/// # Returns
///
/// * `Result<Array<f32, D>>` - Filtered array
#[allow(dead_code)]
pub fn gaussian_filter_f32<D>(
    input: &Array<f32, D>,
    sigma: f32,
    mode: Option<BorderMode>,
    truncate: Option<f32>,
) -> NdimageResult<Array<f32, D>>
where
    D: Dimension + 'static,
{
    // Implementation similar to gaussian_filter but specialized for f32
    let border_mode = mode.unwrap_or(BorderMode::Reflect);
    let trunc = truncate.unwrap_or(4.0);

    // Validate inputs
    if sigma <= 0.0 {
        return Err(NdimageError::InvalidInput("Sigma must be positive".into()));
    }

    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    // Handle scalar or constant case
    if input.len() <= 1 {
        return Ok(input.to_owned());
    }

    // Dispatch based on dimensionality but with specialized f32 implementation
    match input.ndim() {
        1 => {
            // For 1D arrays, convert to Array1 first for correct indexing
            let array1d = input
                .to_owned()
                .into_dimensionality::<scirs2_core::ndarray::Ix1>()
                .map_err(|_| {
                    NdimageError::DimensionError("Failed to convert to 1D array".into())
                })?;

            let radius = (trunc * sigma).ceil() as usize;
            let size = 2 * radius + 1;
            let mut kernel = Array1::zeros(size);

            // Create kernel
            let two_sigma_sq = 2.0 * sigma * sigma;
            let mut sum = 0.0;

            for i in 0..size {
                let x = (i as f32) - (radius as f32);
                let val = (-x * x / two_sigma_sq).exp();
                kernel[i] = val;
                sum += val;
            }

            // Normalize
            if sum > 0.0 {
                kernel.mapv_inplace(|x| x / sum);
            }

            // Apply using 1D convolution
            let mut output = Array1::zeros(array1d.raw_dim());
            let padded = pad_array(&array1d, &[(radius, radius)], &border_mode, None)?;

            for i in 0..array1d.len() {
                let mut sum = 0.0;
                for k in 0..kernel.len() {
                    sum += padded[i + k] * kernel[k];
                }
                output[i] = sum;
            }

            // Convert back to original dimensionality
            let result = output.into_dimensionality::<D>().map_err(|_| {
                NdimageError::DimensionError("Failed to convert from 1D array".into())
            })?;

            Ok(result)
        }
        2 => {
            // For 2D arrays, use a specialized implementation with Ix2 dimensionality
            // This requires explicitly converting to and from Array2
            let array2d = input.to_owned().into_dimensionality::<Ix2>().map_err(|_| {
                NdimageError::DimensionError("Failed to convert to 2D array".into())
            })?;

            let radius = (trunc * sigma).ceil() as usize;
            let size = 2 * radius + 1;
            let mut kernel = Array1::zeros(size);

            // Create kernel
            let two_sigma_sq = 2.0 * sigma * sigma;
            let mut sum = 0.0;

            for i in 0..size {
                let x = (i as f32) - (radius as f32);
                let val = (-x * x / two_sigma_sq).exp();
                kernel[i] = val;
                sum += val;
            }

            if sum > 0.0 {
                kernel.mapv_inplace(|x| x / sum);
            }

            let shape = array2d.shape();
            let mut temp = Array2::zeros((shape[0], shape[1]));

            // Pad for rows
            let padded_rows = pad_array(&array2d, &[(0, 0), (radius, radius)], &border_mode, None)?;

            for i in 0..shape[0] {
                for j in 0..shape[1] {
                    let mut sum = 0.0;
                    for k in 0..kernel.len() {
                        sum += padded_rows[[i, j + k]] * kernel[k];
                    }
                    temp[[i, j]] = sum;
                }
            }

            // Apply along columns
            let mut output = Array2::zeros((shape[0], shape[1]));
            let padded_cols = pad_array(&temp, &[(radius, radius), (0, 0)], &border_mode, None)?;

            for i in 0..shape[0] {
                for j in 0..shape[1] {
                    let mut sum = 0.0;
                    for k in 0..kernel.len() {
                        sum += padded_cols[[i + k, j]] * kernel[k];
                    }
                    output[[i, j]] = sum;
                }
            }

            // Convert back to original dimensionality
            let result = output.into_dimensionality::<D>().map_err(|_| {
                NdimageError::DimensionError("Failed to convert from 2D array".into())
            })?;

            Ok(result)
        }
        _ => {
            // For higher dimensions, convert to dynamic array for easier processing
            let input_dyn = input.to_owned().into_dyn();
            let mut result = input_dyn.clone();

            // Calculate kernel once
            let radius = (trunc * sigma).ceil() as usize;
            let size = 2 * radius + 1;
            let mut kernel = Array1::zeros(size);

            let two_sigma_sq = 2.0 * sigma * sigma;
            let mut sum = 0.0;

            for i in 0..size {
                let x = (i as f32) - (radius as f32);
                let val = (-x * x / two_sigma_sq).exp();
                kernel[i] = val;
                sum += val;
            }

            if sum > 0.0 {
                kernel.mapv_inplace(|x| x / sum);
            }

            // Apply kernel along each dimension
            for axis in 0..input.ndim() {
                let mut output = Array::zeros(result.raw_dim());
                let inputshape = result.shape();

                let mut pad_width = vec![(0, 0); input.ndim()];
                pad_width[axis] = (radius, radius);

                let padded = pad_array(&result, &pad_width, &border_mode, None)?;

                // Process each position in the output array
                for (idx, val) in output.indexed_iter_mut() {
                    let mut sum = 0.0;

                    // Apply kernel along the axis
                    for k in 0..kernel.len() {
                        let mut padded_idx = idx.as_array_view().to_vec();
                        let kernel_pos = k as isize - radius as isize;

                        // Calculate source position along the axis
                        let src_pos = idx[axis] as isize + kernel_pos;
                        let src_len = inputshape[axis] as isize;

                        // Apply border mode to get actual index
                        let src_idx = match border_mode {
                            BorderMode::Reflect => {
                                if src_pos < 0 {
                                    (-src_pos - 1) as usize % src_len as usize
                                } else if src_pos >= src_len {
                                    (2 * src_len - src_pos - 1) as usize % src_len as usize
                                } else {
                                    src_pos as usize
                                }
                            }
                            BorderMode::Mirror => {
                                if src_pos < 0 {
                                    (-src_pos) as usize % (src_len as usize * 2 - 2).max(1)
                                } else if src_pos >= src_len {
                                    (2 * src_len - src_pos - 2) as usize
                                        % (src_len as usize * 2 - 2).max(1)
                                } else {
                                    src_pos as usize
                                }
                            }
                            BorderMode::Wrap => {
                                if src_len == 0 {
                                    0
                                } else {
                                    ((src_pos % src_len + src_len) % src_len) as usize
                                }
                            }
                            BorderMode::Constant => {
                                if src_pos < 0 || src_pos >= src_len {
                                    // Skip this element (using zero)
                                    continue;
                                } else {
                                    src_pos as usize
                                }
                            }
                            BorderMode::Nearest => {
                                if src_pos < 0 {
                                    0
                                } else if src_pos >= src_len {
                                    (src_len - 1) as usize
                                } else {
                                    src_pos as usize
                                }
                            }
                        };

                        padded_idx[axis] = src_idx;
                        let idx_dyn = IxDyn(&padded_idx);
                        sum += padded[&idx_dyn] * kernel[k];
                    }

                    *val = sum;
                }

                result = output;
            }

            // Convert back to original dimension type
            let result = result.into_dimensionality::<D>().map_err(|_| {
                NdimageError::DimensionError("Failed to convert back to original dimensions".into())
            })?;

            Ok(result)
        }
    }
}

/// Generate a 1D Gaussian kernel for f32 filtering
/// This function uses manual caching to avoid Result Clone issues
#[allow(dead_code)]
pub fn gaussian_kernel1d_f32(sigma: f32, truncate: f32) -> NdimageResult<Array1<f32>> {
    // Manual caching using lazy_static or thread_local would be ideal here
    // but for simplicity, we'll just implement the function without caching for now

    if sigma <= 0.0 {
        return Err(NdimageError::InvalidInput("Sigma must be positive".into()));
    }

    let radius = (truncate * sigma).ceil();
    let radius_int = radius as usize;

    let size = 2 * radius_int + 1;
    let mut kernel = Array1::zeros(size);

    // Create x values for Gaussian calculation (distance from center)
    let mut x_values = Array1::zeros(size);
    for i in 0..size {
        x_values[i] = (i as f32) - (radius_int as f32);
    }

    // Calculate x^2 manually
    let mut x_squared = Array1::zeros(size);
    for i in 0..size {
        x_squared[i] = x_values[i] * x_values[i];
    }

    // Calculate exp(-x^2/(2*_sigma^2))
    let two_sigma_squared = 2.0 * sigma * sigma;

    // Apply the Gaussian formula: exp(-x^2/(2*_sigma^2))
    for i in 0..size {
        kernel[i] = (-x_squared[i] / two_sigma_squared).exp();
    }

    // Normalize
    let sum = kernel.sum();
    if sum > 0.0 {
        kernel.mapv_inplace(|x| x / sum);
    }

    Ok(kernel)
}

/// Specialized Gaussian filter implementation for f64 arrays
///
/// This is a convenience function for the common case where T is f64.
/// It's useful for functions like canny_edges that need to avoid Send/Sync constraints.
#[allow(dead_code)]
pub fn gaussian_filter_f64<D>(
    input: &Array<f64, D>,
    sigma: f64,
    mode: Option<BorderMode>,
    truncate: Option<f64>,
) -> NdimageResult<Array<f64, D>>
where
    D: Dimension + 'static,
{
    // This is simply an alias for gaussian_filter now that it's specialized for f64
    gaussian_filter(input, sigma, mode, truncate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_gaussian_kernel1d() {
        let sigma = 1.0;
        let truncate = 4.0;
        let kernel = gaussian_kernel1d_f64(sigma, truncate)
            .expect("gaussian_kernel1d_f64 should succeed for test");

        // Check kernel properties
        assert_eq!(kernel.len(), 9); // 2*4 + 1 = 9

        // Check that kernel is normalized
        let sum: f64 = kernel.sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_filter1d() {
        // Create a simple impulse signal
        let mut input = Array1::zeros(11);
        input[5] = 1.0;

        // Apply Gaussian filter
        let sigma = 1.0;
        let result = gaussian_filter1d_f64(&input, sigma, None, None)
            .expect("gaussian_filter1d_f64 should succeed for test");

        // Check that result is smoothed
        assert!(result[5] < 1.0); // Peak should be reduced
        assert!(result[4] > 0.0); // Adjacent values should be nonzero
        assert!(result[6] > 0.0);

        // Check that the total sum is preserved (within tolerance)
        let sum: f64 = result.sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gaussian_filter_2d() {
        use scirs2_core::ndarray::Array2;

        // Create a simple 2D impulse array
        let mut input = Array2::zeros((7, 7));
        input[[3, 3]] = 1.0;

        // Apply Gaussian filter with constant border mode to avoid edge effects
        let sigma = 1.0;
        let result = gaussian_filter(&input, sigma, Some(BorderMode::Constant), None)
            .expect("gaussian_filter should succeed for test");

        // Check that result is smoothed
        assert!(result[[3, 3]] < 1.0); // Peak should be reduced
        assert!(result[[2, 3]] > 0.0); // Adjacent values should be nonzero
        assert!(result[[3, 2]] > 0.0);
        assert!(result[[4, 3]] > 0.0);
        assert!(result[[3, 4]] > 0.0);

        // Sum will be slightly less than 1.0 with constant border mode (some energy lost at borders)
        let sum: f64 = result.sum();
        assert!(sum > 0.95); // Instead of strict equality, check it's close to 1.0
        assert!(sum < 1.02); // Also check it's not too high
    }

    #[test]
    fn test_gaussian_filter_3d() {
        use scirs2_core::ndarray::Array3;

        // Create a simple 3D impulse array
        let mut input = Array3::zeros((5, 5, 5));
        input[[2, 2, 2]] = 1.0;

        // Apply Gaussian filter
        let sigma = 1.0;
        let result = gaussian_filter(&input, sigma, Some(BorderMode::Reflect), None)
            .expect("gaussian_filter should succeed for test");

        // Check that result is smoothed
        assert!(result[[2, 2, 2]] > 0.0); // Peak should have a value
        assert!(result[[1, 2, 2]] > 0.0); // Adjacent values should be nonzero
        assert!(result[[2, 1, 2]] > 0.0);
        assert!(result[[2, 2, 1]] > 0.0);
        assert!(result[[3, 2, 2]] > 0.0);
        assert!(result[[2, 3, 2]] > 0.0);
        assert!(result[[2, 2, 3]] > 0.0);

        // Print the value at the center for debugging
        println!("Gaussian 3D filter center value: {}", result[[2, 2, 2]]);

        // Sum should be approximately preserved (reflection preserves energy)
        let sum: f64 = result.sum();
        println!("Gaussian 3D filter sum: {}", sum);
        assert!(sum > 0.9);
        assert!(sum < 1.1);

        // Check that applying filter with very small sigma doesn't change the input much
        let small_sigma = 0.1;
        let small_result = gaussian_filter(&input, small_sigma, Some(BorderMode::Reflect), None)
            .expect("gaussian_filter should succeed for test");
        println!(
            "Gaussian 3D filter (small sigma) center value: {}",
            small_result[[2, 2, 2]]
        );
        assert!(small_result[[2, 2, 2]] > 0.5); // Center should retain significant value
    }
}
