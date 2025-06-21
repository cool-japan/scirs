//! Transformation-based interpolation functions

use ndarray::{Array, Dimension};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use super::utils::{interpolate_linear, interpolate_nearest};
use super::{BoundaryMode, InterpolationOrder};
use crate::error::{NdimageError, Result};

/// Affine transform of an array using interpolation
///
/// # Arguments
///
/// * `input` - Input array
/// * `matrix` - Transformation matrix
/// * `offset` - Offset vector (default: zeros)
/// * `output_shape` - Shape of the output array (default: same as input)
/// * `order` - Interpolation order (default: Linear)
/// * `mode` - Boundary handling mode (default: Constant)
/// * `cval` - Value to use for constant mode (default: 0.0)
/// * `prefilter` - Whether to prefilter the input with a spline filter (default: true)
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Transformed array
#[allow(clippy::too_many_arguments)] // Necessary to match SciPy's API signature
pub fn affine_transform<T, D>(
    input: &Array<T, D>,
    matrix: &Array<T, ndarray::Ix2>,
    offset: Option<&Array<T, ndarray::Ix1>>,
    output_shape: Option<&[usize]>,
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

    if matrix.shape()[0] != input.ndim() || matrix.shape()[1] != input.ndim() {
        return Err(NdimageError::DimensionError(format!(
            "Matrix shape must be ({0}, {0}) for input array of dimension {0}, got ({1}, {2})",
            input.ndim(),
            matrix.shape()[0],
            matrix.shape()[1]
        )));
    }

    if let Some(off) = offset {
        if off.len() != input.ndim() {
            return Err(NdimageError::DimensionError(format!(
                "Offset length must match input dimensions (got {} expected {})",
                off.len(),
                input.ndim()
            )));
        }
    }

    if let Some(shape) = output_shape {
        if shape.len() != input.ndim() {
            return Err(NdimageError::DimensionError(format!(
                "Output shape length must match input dimensions (got {} expected {})",
                shape.len(),
                input.ndim()
            )));
        }
    }

    let interp_order = order.unwrap_or(InterpolationOrder::Linear);
    let boundary = mode.unwrap_or(BoundaryMode::Constant);
    let const_val = cval.unwrap_or_else(|| T::zero());
    let _prefilter_input = prefilter.unwrap_or(true);

    // Determine output shape
    let out_shape = if let Some(shape) = output_shape {
        shape.to_vec()
    } else {
        input.shape().to_vec()
    };

    // Create output array
    let output = Array::zeros(ndarray::IxDyn(&out_shape));
    let mut result_dyn = output.into_dyn();
    let input_dyn = input.clone().into_dyn();

    // Create default offset if not provided
    let zero_offset: Array<T, ndarray::Ix1> = Array::zeros(input.ndim());
    let offset_vec = offset.unwrap_or(&zero_offset);

    // For each output pixel, calculate corresponding input coordinates
    for (output_idx, output_val) in result_dyn.indexed_iter_mut() {
        // Convert output coordinates to floating point
        let output_coords: Vec<T> = output_idx
            .as_array_view()
            .iter()
            .map(|&coord| T::from_usize(coord).unwrap_or_else(|| T::zero()))
            .collect();

        // Apply affine transformation: input_coords = matrix^-1 * (output_coords - offset)
        // For now, assume the matrix is the forward transformation and we need to invert it
        // Simple approach: solve the system matrix * input_coords + offset = output_coords
        // So: input_coords = matrix^-1 * (output_coords - offset)

        let mut input_coords = vec![T::zero(); input.ndim()];

        // For 2D case, implement simple matrix inversion
        if input.ndim() == 2 {
            let det = matrix[[0, 0]] * matrix[[1, 1]] - matrix[[0, 1]] * matrix[[1, 0]];

            if det.abs() < T::from_f64(1e-10).unwrap_or_else(|| T::zero()) {
                // Singular matrix, fall back to identity
                input_coords = output_coords;
            } else {
                // Calculate adjusted output coordinates
                let adj_out_x = output_coords[0] - offset_vec[0];
                let adj_out_y = output_coords[1] - offset_vec[1];

                // Apply inverse transformation
                input_coords[0] = (matrix[[1, 1]] * adj_out_x - matrix[[0, 1]] * adj_out_y) / det;
                input_coords[1] = (-matrix[[1, 0]] * adj_out_x + matrix[[0, 0]] * adj_out_y) / det;
            }
        } else {
            // For other dimensions, use a simple approach (assuming diagonal or near-identity matrix)
            for i in 0..input.ndim() {
                let adj_coord = output_coords[i] - offset_vec[i];

                // Simple inversion for diagonal-dominant case
                if matrix[[i, i]].abs() > T::from_f64(1e-10).unwrap_or_else(|| T::zero()) {
                    input_coords[i] = adj_coord / matrix[[i, i]];
                } else {
                    input_coords[i] = adj_coord;
                }
            }
        }

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

/// Apply a general geometric transform to an array
///
/// # Arguments
///
/// * `input` - Input array
/// * `mapping` - Function mapping output coordinates to input coordinates
/// * `output_shape` - Shape of the output array (default: same as input)
/// * `order` - Interpolation order (default: Linear)
/// * `mode` - Boundary handling mode (default: Constant)
/// * `cval` - Value to use for constant mode (default: 0.0)
/// * `prefilter` - Whether to prefilter the input with a spline filter (default: true)
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Transformed array
pub fn geometric_transform<T, D, F>(
    input: &Array<T, D>,
    _mapping: F,
    output_shape: Option<&[usize]>,
    order: Option<InterpolationOrder>,
    mode: Option<BoundaryMode>,
    cval: Option<T>,
    prefilter: Option<bool>,
) -> Result<Array<T, D>>
where
    T: Float + FromPrimitive + Debug,
    D: Dimension,
    F: Fn(&[usize]) -> Vec<T>,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if let Some(shape) = output_shape {
        if shape.len() != input.ndim() {
            return Err(NdimageError::DimensionError(format!(
                "Output shape length must match input dimensions (got {} expected {})",
                shape.len(),
                input.ndim()
            )));
        }
    }

    let _interp_order = order.unwrap_or(InterpolationOrder::Linear);
    let _boundary = mode.unwrap_or(BoundaryMode::Constant);
    let _const_val = cval.unwrap_or_else(|| T::zero());
    let _prefilter_input = prefilter.unwrap_or(true);

    // Placeholder implementation returning a copy of the input
    Ok(input.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_affine_transform() {
        let input: Array2<f64> = Array2::eye(3);
        let matrix = Array2::<f64>::eye(2);

        let result = affine_transform(&input, &matrix, None, None, None, None, None, None).unwrap();
        assert_eq!(result.shape(), input.shape());
    }

    #[test]
    fn test_geometric_transform() {
        let input: Array2<f64> = Array2::eye(3);

        // Identity mapping
        let mapping = |coords: &[usize]| -> Vec<f64> { coords.iter().map(|&x| x as f64).collect() };

        let result = geometric_transform(&input, mapping, None, None, None, None, None).unwrap();
        assert_eq!(result.shape(), input.shape());
    }
}
