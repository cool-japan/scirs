//! Geometric transformation functions

use ndarray::{Array, Dimension};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use super::utils::{interpolate_linear, interpolate_nearest};
use super::{BoundaryMode, InterpolationOrder};
use crate::error::{NdimageError, Result};

/// Zoom an array using interpolation
///
/// # Arguments
///
/// * `input` - Input array
/// * `zoom_factor` - Zoom factor for all dimensions or per dimension
/// * `order` - Interpolation order (default: Linear)
/// * `mode` - Boundary handling mode (default: Constant)
/// * `cval` - Value to use for constant mode (default: 0.0)
/// * `prefilter` - Whether to prefilter the input with a spline filter (default: true)
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Zoomed array
pub fn zoom<T, D>(
    input: &Array<T, D>,
    zoom_factor: T,
    order: Option<InterpolationOrder>,
    mode: Option<BoundaryMode>,
    cval: Option<T>,
    prefilter: Option<bool>,
) -> Result<Array<T, D>>
where
    T: Float + FromPrimitive + Debug,
    D: Dimension,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if zoom_factor <= T::zero() {
        return Err(NdimageError::InvalidInput(format!(
            "Zoom factor must be positive, got {:?}",
            zoom_factor
        )));
    }

    let interp_order = order.unwrap_or(InterpolationOrder::Linear);
    let boundary = mode.unwrap_or(BoundaryMode::Constant);
    let const_val = cval.unwrap_or_else(|| T::zero());
    let _prefilter_input = prefilter.unwrap_or(true);

    // Calculate output shape - each dimension scaled by zoom factor
    let input_shape = input.shape();
    let output_shape: Vec<usize> = input_shape
        .iter()
        .map(|&dim| {
            let new_dim = T::from_usize(dim).unwrap_or_else(|| T::one()) * zoom_factor;
            new_dim.to_usize().unwrap_or(dim)
        })
        .collect();

    // Create output array with calculated shape
    let output = Array::zeros(ndarray::IxDyn(&output_shape));

    // Convert input to dynamic for easier indexing
    let input_dyn = input.clone().into_dyn();

    // Perform interpolation for each output pixel
    let result = output;

    // Convert to dynamic for easier indexing
    let mut result_dyn = result.into_dyn();

    for (output_idx, output_val) in result_dyn.indexed_iter_mut() {
        // Map output coordinates back to input coordinates
        let input_coords: Vec<T> = output_idx
            .as_array_view()
            .iter()
            .map(|&out_coord| {
                // Scale back by zoom factor to get input coordinate
                T::from_usize(out_coord).unwrap_or_else(|| T::zero()) / zoom_factor
            })
            .collect();

        // Perform interpolation
        let interpolated_value = match interp_order {
            InterpolationOrder::Nearest => {
                interpolate_nearest(&input_dyn, &input_coords, &boundary, const_val)
            }
            InterpolationOrder::Linear => {
                interpolate_linear(&input_dyn, &input_coords, &boundary, const_val)
            }
            _ => {
                // For now, fall back to linear for unsupported orders
                interpolate_linear(&input_dyn, &input_coords, &boundary, const_val)
            }
        };

        *output_val = interpolated_value;
    }

    // Convert back to original dimensionality
    result_dyn.into_dimensionality::<D>().map_err(|_| {
        NdimageError::DimensionError("Failed to convert back to original dimensions".into())
    })
}

/// Shift an array using interpolation
///
/// # Arguments
///
/// * `input` - Input array
/// * `shift` - Shift along each dimension
/// * `order` - Interpolation order (default: Linear)
/// * `mode` - Boundary handling mode (default: Constant)
/// * `cval` - Value to use for constant mode (default: 0.0)
/// * `prefilter` - Whether to prefilter the input with a spline filter (default: true)
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Shifted array
pub fn shift<T, D>(
    input: &Array<T, D>,
    shift: &[T],
    order: Option<InterpolationOrder>,
    mode: Option<BoundaryMode>,
    cval: Option<T>,
    prefilter: Option<bool>,
) -> Result<Array<T, D>>
where
    T: Float + FromPrimitive + Debug,
    D: Dimension,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if shift.len() != input.ndim() {
        return Err(NdimageError::DimensionError(format!(
            "Shift length must match input dimensions (got {} expected {})",
            shift.len(),
            input.ndim()
        )));
    }

    let interp_order = order.unwrap_or(InterpolationOrder::Linear);
    let boundary = mode.unwrap_or(BoundaryMode::Constant);
    let const_val = cval.unwrap_or_else(|| T::zero());
    let _prefilter_input = prefilter.unwrap_or(true);

    // Create output array with same shape as input
    let output = Array::zeros(input.raw_dim());

    // Convert input to dynamic for easier indexing
    let input_dyn = input.clone().into_dyn();

    // Perform interpolation for each output pixel
    let result = output;

    // Convert to dynamic for easier indexing
    let mut result_dyn = result.into_dyn();

    for (output_idx, output_val) in result_dyn.indexed_iter_mut() {
        // Map output coordinates back to input coordinates by subtracting shift
        let input_coords: Vec<T> = output_idx
            .as_array_view()
            .iter()
            .enumerate()
            .map(|(i, &out_coord)| {
                // Subtract shift to get input coordinate
                T::from_usize(out_coord).unwrap_or_else(|| T::zero()) - shift[i]
            })
            .collect();

        // Perform interpolation
        let interpolated_value = match interp_order {
            InterpolationOrder::Nearest => {
                interpolate_nearest(&input_dyn, &input_coords, &boundary, const_val)
            }
            InterpolationOrder::Linear => {
                interpolate_linear(&input_dyn, &input_coords, &boundary, const_val)
            }
            _ => {
                // For now, fall back to linear for unsupported orders
                interpolate_linear(&input_dyn, &input_coords, &boundary, const_val)
            }
        };

        *output_val = interpolated_value;
    }

    // Convert back to original dimensionality
    result_dyn.into_dimensionality::<D>().map_err(|e| {
        NdimageError::DimensionError(format!(
            "Failed to convert back to original dimensions: {}",
            e
        ))
    })
}

/// Rotate an array using interpolation
///
/// # Arguments
///
/// * `input` - Input array
/// * `angle` - Rotation angle in degrees
/// * `axes` - Axes of rotation (default: (0, 1))
/// * `reshape` - Whether to reshape the output to contain the full rotated input (default: true)
/// * `order` - Interpolation order (default: Linear)
/// * `mode` - Boundary handling mode (default: Constant)
/// * `cval` - Value to use for constant mode (default: 0.0)
/// * `prefilter` - Whether to prefilter the input with a spline filter (default: true)
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Rotated array
#[allow(clippy::too_many_arguments)] // Necessary to match SciPy's API signature
pub fn rotate<T, D>(
    input: &Array<T, D>,
    angle: T,
    axes: Option<(usize, usize)>,
    reshape: Option<bool>,
    order: Option<InterpolationOrder>,
    mode: Option<BoundaryMode>,
    cval: Option<T>,
    prefilter: Option<bool>,
) -> Result<Array<T, D>>
where
    T: Float + FromPrimitive + Debug,
    D: Dimension,
{
    // Validate inputs
    if input.ndim() < 2 {
        return Err(NdimageError::InvalidInput(
            "Input array must have at least 2 dimensions for rotation".into(),
        ));
    }

    let (axis1, axis2) = axes.unwrap_or((0, 1));

    if axis1 >= input.ndim() || axis2 >= input.ndim() || axis1 == axis2 {
        return Err(NdimageError::InvalidInput(format!(
            "Invalid axes: ({}, {})",
            axis1, axis2
        )));
    }

    let _do_reshape = reshape.unwrap_or(true);
    let interp_order = order.unwrap_or(InterpolationOrder::Linear);
    let boundary = mode.unwrap_or(BoundaryMode::Constant);
    let const_val = cval.unwrap_or_else(|| T::zero());
    let _prefilter_input = prefilter.unwrap_or(true);

    // Convert angle from degrees to radians
    let pi = T::from_f64(std::f64::consts::PI).unwrap_or_else(|| T::one());
    let angle_rad = angle * pi / T::from_f64(180.0).unwrap_or_else(|| T::one());

    // Calculate rotation matrix
    let cos_theta = angle_rad.cos();
    let sin_theta = angle_rad.sin();

    // Create output array with same shape as input (not implementing reshape for now)
    let output = Array::zeros(input.raw_dim());
    let mut result_dyn = output.into_dyn();
    let input_dyn = input.clone().into_dyn();

    // Get input shape and calculate center
    let input_shape = input.shape();
    let center1 =
        T::from_usize(input_shape[axis1]).unwrap_or_else(|| T::one()) / (T::one() + T::one());
    let center2 =
        T::from_usize(input_shape[axis2]).unwrap_or_else(|| T::one()) / (T::one() + T::one());

    // For each output pixel, calculate corresponding input coordinates
    for (output_idx, output_val) in result_dyn.indexed_iter_mut() {
        let mut input_coords: Vec<T> = output_idx
            .as_array_view()
            .iter()
            .map(|&coord| T::from_usize(coord).unwrap_or_else(|| T::zero()))
            .collect();

        // Get output coordinates for rotation axes
        let out_coord1 = input_coords[axis1];
        let out_coord2 = input_coords[axis2];

        // Translate to center, rotate, then translate back
        let centered1 = out_coord1 - center1;
        let centered2 = out_coord2 - center2;

        // Apply rotation matrix (inverse rotation to map output to input)
        let rotated1 = centered1 * cos_theta + centered2 * sin_theta;
        let rotated2 = -centered1 * sin_theta + centered2 * cos_theta;

        // Translate back from center
        input_coords[axis1] = rotated1 + center1;
        input_coords[axis2] = rotated2 + center2;

        // Perform interpolation
        let interpolated_value = match interp_order {
            InterpolationOrder::Nearest => {
                interpolate_nearest(&input_dyn, &input_coords, &boundary, const_val)
            }
            InterpolationOrder::Linear => {
                interpolate_linear(&input_dyn, &input_coords, &boundary, const_val)
            }
            _ => {
                // For now, fall back to linear for unsupported orders
                interpolate_linear(&input_dyn, &input_coords, &boundary, const_val)
            }
        };

        *output_val = interpolated_value;
    }

    // Convert back to original dimensionality
    result_dyn.into_dimensionality::<D>().map_err(|_| {
        NdimageError::DimensionError("Failed to convert back to original dimensions".into())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_zoom() {
        let input: Array2<f64> = Array2::eye(3);
        let result = zoom(&input, 2.0, None, None, None, None).unwrap();
        // With zoom factor 2.0, output shape should be doubled
        assert_eq!(result.shape(), &[6, 6]);

        // Test zooming down
        let result_small = zoom(&input, 0.5, None, None, None, None).unwrap();
        // With zoom factor 0.5, output shape should be halved (3 * 0.5 = 1)
        assert_eq!(result_small.shape(), &[1, 1]);
    }

    #[test]
    fn test_shift_function() {
        let input: Array2<f64> = Array2::eye(3);
        let shift_values = vec![1.0, -1.0];
        let result = shift(&input, &shift_values, None, None, None, None).unwrap();
        assert_eq!(result.shape(), input.shape());

        // Test simple shift with a known pattern
        let mut simple_input = Array2::zeros((3, 3));
        simple_input[[1, 1]] = 1.0; // Center pixel set to 1

        // Shift by [0, 1] should move the center pixel to the right
        let shift_right = vec![0.0, 1.0];
        let result_right = shift(&simple_input, &shift_right, None, None, None, None).unwrap();

        // The pixel should now be at [1, 2] (shifted right)
        // Since we only have a 3x3 array, check if the pixel moved correctly
        assert_eq!(result_right[[1, 2]], 1.0);
    }

    #[test]
    fn test_rotate() {
        let input: Array2<f64> = Array2::eye(3);
        let result = rotate(&input, 45.0, None, None, None, None, None, None).unwrap();
        assert_eq!(result.shape(), input.shape());
    }
}
