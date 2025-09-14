//! Convolution functions for n-dimensional arrays

use ndarray::{Array, Array1, Dimension};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use super::{convolve_optimized as convolve_opt, BorderMode};
use crate::error::{NdimageError, NdimageResult};

/// Apply a uniform filter (box filter or moving average) to an n-dimensional array
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
pub fn uniform_filter<T, D>(
    input: &Array<T, D>,
    size: &[usize],
    mode: Option<BorderMode>,
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
    let _border_mode = mode.unwrap_or(BorderMode::Reflect);

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

    // Delegate to the full uniform filter implementation
    use super::uniform::uniform_filter as uniform_filter_impl;

    uniform_filter_impl(input, size, Some(_border_mode), None)
}

/// Convolve an n-dimensional array with a filter kernel
///
/// # Arguments
///
/// * `input` - Input array to convolve
/// * `weights` - Convolution kernel
/// * `mode` - Border handling mode (defaults to Reflect)
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Convolved array
#[allow(dead_code)]
pub fn convolve<T, D, E>(
    input: &Array<T, D>,
    weights: &Array<T, E>,
    mode: Option<BorderMode>,
) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + Debug + std::ops::AddAssign + std::ops::DivAssign + Clone + 'static,
    D: Dimension + 'static,
    E: Dimension + 'static,
{
    let border_mode = mode.unwrap_or(BorderMode::Reflect);

    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if weights.ndim() != input.ndim() {
        return Err(NdimageError::DimensionError(format!(
            "Weights must have same rank as input (got {} expected {})",
            weights.ndim(),
            input.ndim()
        )));
    }

    // For 2D arrays, use specialized implementation
    if input.ndim() == 2 && weights.ndim() == 2 {
        match (input.ndim(), weights.ndim()) {
            (2, 2) => {
                // Convert to 2D arrays
                let input_2d = input
                    .to_owned()
                    .into_dimensionality::<ndarray::Ix2>()
                    .map_err(|_| {
                        NdimageError::DimensionError("Failed to convert to 2D array".into())
                    })?;

                let weights_2d = weights
                    .to_owned()
                    .into_dimensionality::<ndarray::Ix2>()
                    .map_err(|_| {
                        NdimageError::DimensionError("Failed to convert weights to 2D array".into())
                    })?;

                // Call 2D implementation
                let result = convolve_2d(&input_2d, &weights_2d, &border_mode)?;

                // Convert back to original dimensionality
                result.into_dimensionality::<D>().map_err(|_| {
                    NdimageError::DimensionError(
                        "Failed to convert result back to original dimensions".into(),
                    )
                })
            }
            _ => unreachable!(), // Already checked dimensions above
        }
    } else {
        // Handle other dimensionalities (1D, 3D, etc.)
        match (input.ndim(), weights.ndim()) {
            (1, 1) => {
                // 1D convolution
                let input_1d = input
                    .to_owned()
                    .into_dimensionality::<ndarray::Ix1>()
                    .map_err(|_| {
                        NdimageError::DimensionError("Failed to convert to 1D array".into())
                    })?;

                let weights_1d = weights
                    .to_owned()
                    .into_dimensionality::<ndarray::Ix1>()
                    .map_err(|_| {
                        NdimageError::DimensionError("Failed to convert weights to 1D array".into())
                    })?;

                let result = convolve_1d(&input_1d, &weights_1d, &border_mode)?;
                result.into_dimensionality::<D>().map_err(|_| {
                    NdimageError::DimensionError(
                        "Failed to convert result back to original dimensions".into(),
                    )
                })
            }
            (3, 3) => {
                // 3D convolution
                let input_3d = input
                    .to_owned()
                    .into_dimensionality::<ndarray::Ix3>()
                    .map_err(|_| {
                        NdimageError::DimensionError("Failed to convert to 3D array".into())
                    })?;

                let weights_3d = weights
                    .to_owned()
                    .into_dimensionality::<ndarray::Ix3>()
                    .map_err(|_| {
                        NdimageError::DimensionError("Failed to convert weights to 3D array".into())
                    })?;

                let result = convolve_3d(&input_3d, &weights_3d, &border_mode)?;
                result.into_dimensionality::<D>().map_err(|_| {
                    NdimageError::DimensionError(
                        "Failed to convert result back to original dimensions".into(),
                    )
                })
            }
            _n_ => {
                // Generic n-dimensional convolution
                convolve_nd(input, weights, &border_mode)
            }
        }
    }
}

/// Perform 2D convolution with a kernel
#[allow(dead_code)]
fn convolve_2d<T>(
    input: &Array<T, ndarray::Ix2>,
    weights: &Array<T, ndarray::Ix2>,
    mode: &BorderMode,
) -> NdimageResult<Array<T, ndarray::Ix2>>
where
    T: Float + FromPrimitive + Debug + std::ops::AddAssign + std::ops::DivAssign + Clone + 'static,
{
    // Get dimensions
    let (input_rows, input_cols) = input.dim();
    let (weights_rows, weights_cols) = weights.dim();

    // Calculate padding required
    let pad_rows_before = weights_rows / 2;
    let pad_rows_after = weights_rows - pad_rows_before - 1;
    let pad_cols_before = weights_cols / 2;
    let pad_cols_after = weights_cols - pad_cols_before - 1;

    // Create output array
    let mut output = Array::<T, ndarray::Ix2>::zeros((input_rows, input_cols));

    // Create padding configuration
    let pad_width = vec![
        (pad_rows_before, pad_rows_after),
        (pad_cols_before, pad_cols_after),
    ];

    // Import pad_array from parent module
    use super::pad_array;

    // Pad the input array
    let padded = pad_array(input, &pad_width, mode, None)?;

    // Apply convolution
    for i in 0..input_rows {
        for j in 0..input_cols {
            let mut sum = T::zero();

            // Apply the kernel at each position
            for ki in 0..weights_rows {
                for kj in 0..weights_cols {
                    let input_val = padded[[i + ki, j + kj]];
                    let weight = weights[[weights_rows - ki - 1, weights_cols - kj - 1]]; // Flip kernel for convolution
                    sum += input_val * weight;
                }
            }

            output[[i, j]] = sum;
        }
    }

    Ok(output)
}

/// Convolve an n-dimensional array with a filter kernel using optimized boundary handling
///
/// This version uses virtual boundary handling to avoid creating padded arrays,
/// which is more memory-efficient for large arrays.
///
/// # Arguments
///
/// * `input` - Input array to convolve
/// * `weights` - Convolution kernel
/// * `mode` - Border handling mode (defaults to Reflect)
/// * `use_optimization` - Whether to use the optimized implementation
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Convolved array
#[allow(dead_code)]
pub fn convolve_fast<T, D, E>(
    input: &Array<T, D>,
    weights: &Array<T, E>,
    mode: Option<BorderMode>,
    use_optimization: bool,
) -> NdimageResult<Array<T, D>>
where
    T: Float
        + FromPrimitive
        + Debug
        + std::ops::AddAssign
        + std::ops::DivAssign
        + Clone
        + Send
        + Sync
        + 'static,
    D: Dimension + 'static,
    E: Dimension + 'static,
{
    if use_optimization {
        // Use the optimized implementation
        convolve_opt(input, weights, mode.unwrap_or(BorderMode::Reflect), None)
    } else {
        // Use the standard implementation
        convolve(input, weights, mode)
    }
}

/// Apply a 1D correlation along the specified axis
#[allow(dead_code)]
pub fn correlate1d<T, D>(
    input: &Array<T, D>,
    weights: &Array1<T>,
    axis: usize,
    mode: Option<BorderMode>,
    cval: Option<T>,
) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + Debug + Clone + std::ops::AddAssign + std::ops::DivAssign + 'static,
    D: Dimension + 'static,
{
    use ndarray::IxDyn;

    let mode = mode.unwrap_or(BorderMode::Reflect);
    let _cval = cval.unwrap_or(T::zero());

    // Validate axis
    if axis >= input.ndim() {
        return Err(NdimageError::InvalidInput(format!(
            "Axis {} is out of bounds for array of dimension {}",
            axis,
            input.ndim()
        )));
    }

    // Convert to dynamic dimension for easier manipulation
    let input_dyn = input.to_owned().into_dimensionality::<IxDyn>().unwrap();

    // Create padding for this axis only
    let mut pad_width = vec![(0, 0); input.ndim()];
    let kernel_size = weights.len();
    let pad_before = kernel_size / 2;
    let pad_after = kernel_size - pad_before - 1;
    pad_width[axis] = (pad_before, pad_after);

    // Import pad_array
    use super::pad_array;

    // Pad the input
    let padded = pad_array(&input_dyn, &pad_width, &mode, cval)?;

    // Create output array
    let mut output = Array::zeros(input_dyn.raw_dim());

    // Iterate over the output and compute convolution
    for out_idx in ndarray::indices(output.shape()) {
        let mut sum = T::zero();

        // For this output position, compute the 1D convolution
        let out_coords: Vec<_> = out_idx.slice().to_vec();

        for k in 0..kernel_size {
            let mut in_coords = out_coords.clone();
            in_coords[axis] = out_coords[axis] + k;

            sum = sum + padded[IxDyn(&in_coords)] * weights[k];
        }

        output[IxDyn(&out_coords)] = sum;
    }

    // Convert back to the original dimension type
    output.into_dimensionality().map_err(|_| {
        NdimageError::DimensionError("Failed to convert output back to original dimension".into())
    })
}

/// Perform 1D convolution with a kernel
#[allow(dead_code)]
fn convolve_1d<T>(
    input: &Array<T, ndarray::Ix1>,
    weights: &Array<T, ndarray::Ix1>,
    mode: &BorderMode,
) -> NdimageResult<Array<T, ndarray::Ix1>>
where
    T: Float + FromPrimitive + Debug + std::ops::AddAssign + std::ops::DivAssign + Clone + 'static,
{
    let input_len = input.len();
    let weights_len = weights.len();

    if weights_len == 0 {
        return Err(NdimageError::InvalidInput("Kernel cannot be empty".into()));
    }

    let half_kernel = weights_len / 2;
    let mut output = Array::zeros(input_len);

    for i in 0..input_len {
        let mut sum = T::zero();

        for k in 0..weights_len {
            let input_idx = (i as isize) + (k as isize) - (half_kernel as isize);

            // Handle border mode
            let value = if input_idx < 0 || input_idx >= input_len as isize {
                match mode {
                    BorderMode::Constant => T::zero(),
                    BorderMode::Nearest => {
                        if input_idx < 0 {
                            input[0]
                        } else {
                            input[input_len - 1]
                        }
                    }
                    BorderMode::Reflect => {
                        let reflected_idx = if input_idx < 0 {
                            (-input_idx - 1) as usize
                        } else {
                            2 * input_len - 1 - (input_idx as usize)
                        };
                        input[reflected_idx.min(input_len - 1)]
                    }
                    BorderMode::Wrap => {
                        let wrapped_idx = ((input_idx % (input_len as isize) + input_len as isize)
                            % (input_len as isize))
                            as usize;
                        input[wrapped_idx]
                    }
                    BorderMode::Mirror => {
                        let mirrored_idx = if input_idx < 0 {
                            (-input_idx) as usize
                        } else {
                            2 * input_len - 2 - (input_idx as usize)
                        };
                        input[mirrored_idx.min(input_len - 1)]
                    }
                }
            } else {
                input[input_idx as usize]
            };

            sum += value * weights[k];
        }

        output[i] = sum;
    }

    Ok(output)
}

/// Perform 3D convolution with a kernel
#[allow(dead_code)]
fn convolve_3d<T>(
    input: &Array<T, ndarray::Ix3>,
    weights: &Array<T, ndarray::Ix3>,
    mode: &BorderMode,
) -> NdimageResult<Array<T, ndarray::Ix3>>
where
    T: Float + FromPrimitive + Debug + std::ops::AddAssign + std::ops::DivAssign + Clone + 'static,
{
    let (input_d, input_h, input_w) = input.dim();
    let (weights_d, weights_h, weights_w) = weights.dim();

    if weights_d == 0 || weights_h == 0 || weights_w == 0 {
        return Err(NdimageError::InvalidInput(
            "Kernel cannot have zero dimensions".into(),
        ));
    }

    let half_d = weights_d / 2;
    let half_h = weights_h / 2;
    let half_w = weights_w / 2;

    let mut output = Array::zeros((input_d, input_h, input_w));

    for z in 0..input_d {
        for y in 0..input_h {
            for x in 0..input_w {
                let mut sum = T::zero();

                for kz in 0..weights_d {
                    for ky in 0..weights_h {
                        for kx in 0..weights_w {
                            let input_z = (z as isize) + (kz as isize) - (half_d as isize);
                            let input_y = (y as isize) + (ky as isize) - (half_h as isize);
                            let input_x = (x as isize) + (kx as isize) - (half_w as isize);

                            let value = get_padded_value_3d(input, input_z, input_y, input_x, mode);
                            sum += value * weights[[kz, ky, kx]];
                        }
                    }
                }

                output[[z, y, x]] = sum;
            }
        }
    }

    Ok(output)
}

/// Helper function to get padded values for 3D arrays
#[allow(dead_code)]
fn get_padded_value_3d<T>(
    input: &Array<T, ndarray::Ix3>,
    z: isize,
    y: isize,
    x: isize,
    mode: &BorderMode,
) -> T
where
    T: Float + Clone + 'static,
{
    let (depth, height, width) = input.dim();

    if z >= 0 && z < depth as isize && y >= 0 && y < height as isize && x >= 0 && x < width as isize
    {
        return input[[z as usize, y as usize, x as usize]];
    }

    match mode {
        BorderMode::Constant => T::zero(),
        BorderMode::Nearest => {
            let clamped_z = z.max(0).min(depth as isize - 1) as usize;
            let clamped_y = y.max(0).min(height as isize - 1) as usize;
            let clamped_x = x.max(0).min(width as isize - 1) as usize;
            input[[clamped_z, clamped_y, clamped_x]]
        }
        BorderMode::Reflect => {
            let reflect_coord = |coord: isize, size: usize| -> usize {
                if coord < 0 {
                    (-coord - 1) as usize % size
                } else if coord >= size as isize {
                    (2 * size - 1 - coord as usize) % size
                } else {
                    coord as usize
                }
            };

            let ref_z = reflect_coord(z, depth);
            let ref_y = reflect_coord(y, height);
            let ref_x = reflect_coord(x, width);
            input[[ref_z, ref_y, ref_x]]
        }
        BorderMode::Wrap => {
            let wrap_coord = |coord: isize, size: usize| -> usize {
                ((coord % size as isize + size as isize) % size as isize) as usize
            };

            let wrap_z = wrap_coord(z, depth);
            let wrap_y = wrap_coord(y, height);
            let wrap_x = wrap_coord(x, width);
            input[[wrap_z, wrap_y, wrap_x]]
        }
        BorderMode::Mirror => {
            let mirror_coord = |coord: isize, size: usize| -> usize {
                if coord < 0 {
                    (-coord) as usize % size
                } else if coord >= size as isize {
                    (2 * size - 2 - coord as usize) % size
                } else {
                    coord as usize
                }
            };

            let mir_z = mirror_coord(z, depth);
            let mir_y = mirror_coord(y, height);
            let mir_x = mirror_coord(x, width);
            input[[mir_z, mir_y, mir_x]]
        }
    }
}

/// Generic n-dimensional convolution
#[allow(dead_code)]
fn convolve_nd<T, D, E>(
    input: &Array<T, D>,
    weights: &Array<T, E>,
    mode: &BorderMode,
) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + Debug + std::ops::AddAssign + std::ops::DivAssign + Clone + 'static,
    D: Dimension + 'static,
    E: Dimension + 'static,
{
    let inputshape = input.shape();
    let weightsshape = weights.shape();
    let ndim = input.ndim();

    if ndim != weights.ndim() {
        return Err(NdimageError::DimensionError(
            "Input and weights must have the same number of dimensions".into(),
        ));
    }

    if weightsshape.iter().any(|&s| s == 0) {
        return Err(NdimageError::InvalidInput(
            "Kernel cannot have zero dimensions".into(),
        ));
    }

    // Calculate half sizes for centering the kernel
    let half_sizes: Vec<usize> = weightsshape.iter().map(|&s| s / 2).collect();

    let mut output = Array::zeros(input.raw_dim());

    // Iterate over all output positions
    for out_indices in ndarray::indices(inputshape) {
        let out_coords: Vec<usize> = out_indices.slice().to_vec();
        let mut sum = T::zero();

        // Iterate over all kernel positions
        for weight_indices in ndarray::indices(weightsshape) {
            let weight_coords: Vec<usize> = weight_indices.slice().to_vec();

            // Calculate input coordinates
            let mut input_coords = vec![0isize; ndim];
            for d in 0..ndim {
                input_coords[d] = (out_coords[d] as isize) + (weight_coords[d] as isize)
                    - (half_sizes[d] as isize);
            }

            // Get padded value
            let value = get_padded_value_nd(input, &input_coords, mode);
            // Convert weights to dynamic view for safer indexing
            let weights_dyn = weights.view().into_dyn();
            let weight = weights_dyn[weight_indices];
            sum += value * weight;
        }

        // Convert output to dynamic view for safer indexing
        let mut output_dyn = output.view_mut().into_dyn();
        output_dyn[out_indices] = sum;
    }

    Ok(output)
}

/// Helper function to get padded values for n-dimensional arrays
#[allow(dead_code)]
fn get_padded_value_nd<T, D>(input: &Array<T, D>, coords: &[isize], mode: &BorderMode) -> T
where
    T: Float + Clone + 'static,
    D: Dimension + 'static,
{
    let shape = input.shape();
    let ndim = input.ndim();

    // Check if coordinates are within bounds
    let mut in_bounds = true;
    let mut clamped_coords = vec![0usize; ndim];

    for d in 0..ndim {
        if coords[d] < 0 || coords[d] >= shape[d] as isize {
            in_bounds = false;
        }
        clamped_coords[d] = coords[d].max(0).min(shape[d] as isize - 1) as usize;
    }

    if in_bounds {
        // Convert to dynamic dimension for safe indexing
        let input_dyn = input.view().into_dyn();
        return input_dyn[ndarray::IxDyn(&clamped_coords)];
    }

    match mode {
        BorderMode::Constant => T::zero(),
        BorderMode::Nearest => {
            let input_dyn = input.view().into_dyn();
            input_dyn[ndarray::IxDyn(&clamped_coords)]
        }
        BorderMode::Reflect => {
            let mut reflected_coords = vec![0usize; ndim];
            for d in 0..ndim {
                reflected_coords[d] = if coords[d] < 0 {
                    (-coords[d] - 1) as usize % shape[d]
                } else if coords[d] >= shape[d] as isize {
                    (2 * shape[d] - 1 - coords[d] as usize) % shape[d]
                } else {
                    coords[d] as usize
                };
            }
            let input_dyn = input.view().into_dyn();
            input_dyn[ndarray::IxDyn(&reflected_coords)]
        }
        BorderMode::Wrap => {
            let mut wrapped_coords = vec![0usize; ndim];
            for d in 0..ndim {
                wrapped_coords[d] = ((coords[d] % shape[d] as isize + shape[d] as isize)
                    % shape[d] as isize) as usize;
            }
            let input_dyn = input.view().into_dyn();
            input_dyn[ndarray::IxDyn(&wrapped_coords)]
        }
        BorderMode::Mirror => {
            let mut mirrored_coords = vec![0usize; ndim];
            for d in 0..ndim {
                mirrored_coords[d] = if coords[d] < 0 {
                    (-coords[d]) as usize % shape[d]
                } else if coords[d] >= shape[d] as isize {
                    (2 * shape[d] - 2 - coords[d] as usize) % shape[d]
                } else {
                    coords[d] as usize
                };
            }
            let input_dyn = input.view().into_dyn();
            input_dyn[ndarray::IxDyn(&mirrored_coords)]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_uniform_filter() {
        // Create a simple test image
        let image: Array2<f64> = Array2::eye(5);

        // Apply filter
        let result = uniform_filter(&image, &[3, 3], None).unwrap();

        // Check that result has the same shape
        assert_eq!(result.shape(), image.shape());
    }

    #[test]
    fn test_convolve() {
        // Create a simple test image and kernel
        let image: Array2<f64> = Array2::eye(5);
        let kernel: Array2<f64> = Array2::from_elem((3, 3), 1.0 / 9.0);

        // Apply convolution
        let result = convolve(&image, &kernel, None).unwrap();

        // Check that result has the same shape
        assert_eq!(result.shape(), image.shape());
    }
}
