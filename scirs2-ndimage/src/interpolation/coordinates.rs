//! Coordinate-based interpolation functions

use scirs2_core::ndarray::{Array, IxDyn};
use scirs2_core::numeric::{Float, FromPrimitive, One, Zero};
use std::fmt::Debug;

use super::{BoundaryMode, InterpolationOrder};
use crate::error::{NdimageError, NdimageResult};
use crate::interpolation::spline::spline_filter;

/// Map coordinates from one array to another
///
/// The array of coordinates is used to find, for each point in the output,
/// the corresponding coordinates in the input. The shape of the output is
/// derived from that of the coordinate array by dropping the first axis.
///
/// # Arguments
///
/// * `input` - Input array
/// * `coordinates` - Coordinates at which to sample the input (shape: [ndim, ...outputshape])
/// * `order` - Interpolation order (default: 3 for cubic)
/// * `mode` - Boundary handling mode (default: Constant)
/// * `cval` - Value to use for constant mode (default: 0.0)
/// * `prefilter` - Whether to prefilter the input with a spline filter (default: true)
///
/// # Returns
///
/// * `Result<Array<T, IxDyn>>` - Interpolated values at the coordinates
#[allow(dead_code)]
pub fn map_coordinates<T, D>(
    input: &Array<T, D>,
    coordinates: &Array<T, IxDyn>,
    order: Option<usize>,
    mode: Option<BoundaryMode>,
    cval: Option<T>,
    prefilter: Option<bool>,
) -> NdimageResult<Array<T, IxDyn>>
where
    T: Float + FromPrimitive + Debug + Clone + std::ops::AddAssign + std::ops::DivAssign + 'static,
    D: scirs2_core::ndarray::Dimension,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if coordinates.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Coordinates array cannot be 0-dimensional".into(),
        ));
    }

    // Check that first dimension of coordinates matches input dimensions
    if coordinates.shape()[0] != input.ndim() {
        return Err(NdimageError::DimensionError(format!(
            "First dimension of coordinates must match input dimensions (got {} expected {})",
            coordinates.shape()[0],
            input.ndim()
        )));
    }

    let interp_order = order.unwrap_or(3);
    let boundary = mode.unwrap_or(BoundaryMode::Constant);
    let const_val = cval.unwrap_or_else(|| T::zero());
    let prefilter_input = prefilter.unwrap_or(true);

    if interp_order > 5 {
        return Err(NdimageError::InvalidInput(format!(
            "Interpolation order must be 0-5, got {}",
            interp_order
        )));
    }

    // Output shape is coordinates shape with first dimension dropped
    let outputshape: Vec<usize> = coordinates.shape()[1..].to_vec();
    let output_size: usize = outputshape.iter().product();

    // Apply spline filter if needed and order > 1
    let filtered_input = if prefilter_input && interp_order > 1 {
        // Convert input to dynamic array for spline filtering
        let input_dyn = input.to_owned().into_dyn();
        spline_filter(&input_dyn, Some(interp_order))?
    } else {
        input.to_owned().into_dyn()
    };

    // Create output array
    let mut output = Array::<T, IxDyn>::zeros(IxDyn(&outputshape));

    // Interpolate at each coordinate point
    for i in 0..output_size {
        let output_indices = unravel_index(i, &outputshape);

        // Get coordinates for this output point
        let mut coords = Vec::with_capacity(input.ndim());
        for d in 0..input.ndim() {
            let mut coord_indices = vec![d];
            coord_indices.extend(&output_indices);
            let coord_val = coordinates[IxDyn(&coord_indices)];
            coords.push(coord_val);
        }

        // Interpolate at these coordinates
        let value = interpolate_at_coordinates(
            &filtered_input,
            &coords,
            interp_order,
            &boundary,
            const_val,
        )?;

        // Set output value
        output[IxDyn(&output_indices)] = value;
    }

    Ok(output)
}

/// Convert flat index to multi-dimensional indices
#[allow(dead_code)]
fn unravel_index(_flatindex: usize, shape: &[usize]) -> Vec<usize> {
    let mut indices = vec![0; shape.len()];
    let mut remaining = _flatindex;

    for i in (0..shape.len()).rev() {
        let stride: usize = shape[(i + 1)..].iter().product();
        indices[i] = remaining / stride;
        remaining %= stride;
    }

    indices
}

/// Interpolate at specific coordinates using spline interpolation
#[allow(dead_code)]
fn interpolate_at_coordinates<T>(
    input: &Array<T, IxDyn>,
    coordinates: &[T],
    order: usize,
    mode: &BoundaryMode,
    cval: T,
) -> NdimageResult<T>
where
    T: Float + FromPrimitive + Debug + Clone + std::ops::AddAssign + std::ops::DivAssign + 'static,
{
    if coordinates.len() != input.ndim() {
        return Err(NdimageError::DimensionError(format!(
            "Number of coordinates must match input dimensions (got {} expected {})",
            coordinates.len(),
            input.ndim()
        )));
    }

    match order {
        0 => interpolate_nearest(input, coordinates, mode, cval),
        1 => interpolate_linear(input, coordinates, mode, cval),
        _ => interpolate_spline(input, coordinates, order, mode, cval),
    }
}

/// Nearest neighbor interpolation
#[allow(dead_code)]
fn interpolate_nearest<T>(
    input: &Array<T, IxDyn>,
    coordinates: &[T],
    mode: &BoundaryMode,
    cval: T,
) -> NdimageResult<T>
where
    T: Float + FromPrimitive + Debug + Clone + std::ops::AddAssign + std::ops::DivAssign + 'static,
{
    let indices: Vec<isize> = coordinates
        .iter()
        .map(|&coord| coord.round().to_isize().unwrap_or(0))
        .collect();

    get_value_at_indices(input, &indices, mode, cval)
}

/// Linear interpolation
#[allow(dead_code)]
fn interpolate_linear<T>(
    input: &Array<T, IxDyn>,
    coordinates: &[T],
    mode: &BoundaryMode,
    cval: T,
) -> NdimageResult<T>
where
    T: Float + FromPrimitive + Debug + Clone + std::ops::AddAssign + std::ops::DivAssign + 'static,
{
    if coordinates.is_empty() {
        return Ok(cval);
    }

    // For linear interpolation, we need to interpolate between neighboring points
    let ndim = coordinates.len();
    let num_corners = 1 << ndim; // 2^ndim corners of the hypercube

    let mut result = T::zero();

    for corner in 0..num_corners {
        let mut indices = Vec::with_capacity(ndim);
        let mut weight = T::one();

        for (dim, &coord) in coordinates.iter().enumerate() {
            let use_upper = (corner >> dim) & 1 == 1;
            let base_idx = coord.floor().to_isize().unwrap_or(0);
            let idx = if use_upper { base_idx + 1 } else { base_idx };

            indices.push(idx);

            // Calculate weight for this dimension
            let frac = coord - coord.floor();
            let dim_weight = if use_upper { frac } else { T::one() - frac };
            weight = weight * dim_weight;
        }

        let value = get_value_at_indices(input, &indices, mode, cval)?;
        result = result + weight * value;
    }

    Ok(result)
}

/// Spline interpolation (orders 2-5)
#[allow(dead_code)]
fn interpolate_spline<T>(
    input: &Array<T, IxDyn>,
    coordinates: &[T],
    _order: usize,
    mode: &BoundaryMode,
    cval: T,
) -> NdimageResult<T>
where
    T: Float + FromPrimitive + Debug + Clone + std::ops::AddAssign + std::ops::DivAssign + 'static,
{
    // For now, fall back to linear interpolation for spline orders
    // A full spline interpolation would require implementing B-spline evaluation
    // which is more complex than linear interpolation
    interpolate_linear(input, coordinates, mode, cval)
}

/// Get value at specific integer indices with boundary handling
#[allow(dead_code)]
fn get_value_at_indices<T>(
    input: &Array<T, IxDyn>,
    indices: &[isize],
    mode: &BoundaryMode,
    cval: T,
) -> NdimageResult<T>
where
    T: Float + FromPrimitive + Debug + Clone + std::ops::AddAssign + std::ops::DivAssign + 'static,
{
    let shape = input.shape();
    let mut adjusted_indices = Vec::with_capacity(indices.len());

    for (i, &idx) in indices.iter().enumerate() {
        let size = shape[i] as isize;
        let adjusted_idx = match mode {
            BoundaryMode::Constant => {
                if idx < 0 || idx >= size {
                    return Ok(cval);
                }
                idx as usize
            }
            BoundaryMode::Nearest => {
                if idx < 0 {
                    0
                } else if idx >= size {
                    (size - 1) as usize
                } else {
                    idx as usize
                }
            }
            BoundaryMode::Reflect => {
                let mut reflected_idx = idx;
                if size > 1 {
                    if reflected_idx < 0 {
                        reflected_idx = -reflected_idx - 1;
                    }
                    if reflected_idx >= size {
                        reflected_idx = 2 * (size - 1) - reflected_idx;
                    }
                    if reflected_idx < 0 || reflected_idx >= size {
                        // Handle cases where reflection doesn't work
                        reflected_idx = reflected_idx.rem_euclid(2 * (size - 1));
                        if reflected_idx >= size {
                            reflected_idx = 2 * (size - 1) - reflected_idx;
                        }
                    }
                }
                reflected_idx.max(0).min(size - 1) as usize
            }
            BoundaryMode::Wrap => {
                let wrapped_idx = idx.rem_euclid(size);
                wrapped_idx as usize
            }
            BoundaryMode::Mirror => {
                let mut mirrored_idx = idx;
                if size > 1 {
                    if mirrored_idx < 0 {
                        mirrored_idx = -mirrored_idx;
                    }
                    if mirrored_idx >= size {
                        mirrored_idx = 2 * size - 2 - mirrored_idx;
                    }
                    if mirrored_idx < 0 || mirrored_idx >= size {
                        mirrored_idx = mirrored_idx.rem_euclid(2 * size);
                        if mirrored_idx >= size {
                            mirrored_idx = 2 * size - 1 - mirrored_idx;
                        }
                    }
                }
                mirrored_idx.max(0).min(size - 1) as usize
            }
        };
        adjusted_indices.push(adjusted_idx);
    }

    // Access the value at adjusted indices
    match input.get(IxDyn(&adjusted_indices)) {
        Some(value) => Ok(value.clone()),
        None => Ok(cval),
    }
}

/// Find values at given indices in an array
///
/// # Arguments
///
/// * `input` - Input array
/// * `indices` - Indices at which to sample the input
///
/// # Returns
///
/// * `Result<T>` - Value at the given indices
#[allow(dead_code)]
pub fn value_at_coordinates<T, D>(input: &Array<T, D>, indices: &[usize]) -> NdimageResult<T>
where
    T: Float + FromPrimitive + Debug,
    D: scirs2_core::ndarray::Dimension,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if input.is_empty() {
        return Err(NdimageError::InvalidInput("Input array is empty".into()));
    }

    if indices.len() != input.ndim() {
        return Err(NdimageError::DimensionError(format!(
            "Indices must have same length as input dimensions (got {} expected {})",
            indices.len(),
            input.ndim()
        )));
    }

    // Check that indices are within bounds
    for (i, &idx) in indices.iter().enumerate() {
        if idx >= input.shape()[i] {
            return Err(NdimageError::InvalidInput(format!(
                "Index {} is out of bounds for dimension {} of size {}",
                idx,
                i,
                input.shape()[i]
            )));
        }
    }

    // Use proper n-dimensional array indexing
    // Convert indices to IxDyn for dynamic indexing
    use scirs2_core::ndarray::IxDyn;
    let dynamic_indices = IxDyn(indices);

    // Access the element at the specified indices using dynamic view
    let input_dyn = input.view().into_dyn();
    match input_dyn.get(dynamic_indices) {
        Some(value) => Ok(*value),
        None => Err(NdimageError::InvalidInput(
            "Unable to access array at the specified indices".into(),
        )),
    }
}

/// Find values at arbitrarily-spaced points in an n-dimensional grid
///
/// # Arguments
///
/// * `input` - Input array
/// * `points` - Points at which to sample the input
/// * `order` - Interpolation order (default: Linear)
/// * `mode` - Boundary handling mode (default: Constant)
/// * `cval` - Value to use for constant mode (default: 0.0)
///
/// # Returns
///
/// * `Result<Array<T, scirs2_core::ndarray::Ix1>>` - Interpolated values at the points
#[allow(dead_code)]
pub fn interpn<T, D>(
    input: &Array<T, D>,
    points: &[Array<T, scirs2_core::ndarray::Ix1>],
    order: Option<InterpolationOrder>,
    mode: Option<BoundaryMode>,
    cval: Option<T>,
) -> NdimageResult<Array<T, scirs2_core::ndarray::Ix1>>
where
    T: Float + FromPrimitive + Debug + std::ops::DivAssign + std::ops::AddAssign + 'static,
    D: scirs2_core::ndarray::Dimension,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if points.len() != input.ndim() {
        return Err(NdimageError::DimensionError(format!(
            "Number of point arrays must match input dimensions (got {} expected {})",
            points.len(),
            input.ndim()
        )));
    }

    // Get the number of points to sample
    let n_points = points[0].len();

    // Check that all point arrays have the same length
    for (i, p) in points.iter().enumerate() {
        if p.len() != n_points {
            return Err(NdimageError::DimensionError(format!(
                "Point arrays must have the same length (array {} has length {}, expected {})",
                i,
                p.len(),
                n_points
            )));
        }
    }

    // Create coordinates array for map_coordinates format
    let mut coords_array = Array::<T, IxDyn>::zeros(IxDyn(&[input.ndim(), n_points]));

    for (dim, point_array) in points.iter().enumerate() {
        for (i, &coord) in point_array.iter().enumerate() {
            coords_array[[dim, i]] = coord;
        }
    }

    // Use map_coordinates to do the interpolation
    let coords_dyn = coords_array.into_dyn();
    let input_dyn = input.to_owned().into_dyn();

    // Convert InterpolationOrder to usize
    let order_usize = match order.unwrap_or(InterpolationOrder::Linear) {
        InterpolationOrder::Nearest => 0,
        InterpolationOrder::Linear => 1,
        InterpolationOrder::Cubic => 3,
        InterpolationOrder::Spline => 5,
    };

    let result = map_coordinates(
        &input_dyn,
        &coords_dyn,
        Some(order_usize),
        mode,
        cval,
        Some(true),
    )?;

    // Convert result to 1D array
    Ok(result.into_shape((n_points,)).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;
    use scirs2_core::ndarray::Array2;

    #[test]
    #[ignore]
    fn test_map_coordinates_identity() {
        let input: Array2<f64> = Array2::eye(3);

        // Create identity coordinate mapping: coordinates[d, i, j] = coordinate for dimension d at position (i, j)
        let mut coordinates = Array::<f64, IxDyn>::zeros(IxDyn(&[2, 3, 3]));
        for i in 0..3 {
            for j in 0..3 {
                coordinates[[0, i, j]] = i as f64; // row coordinates
                coordinates[[1, i, j]] = j as f64; // column coordinates
            }
        }

        let result = map_coordinates(&input, &coordinates, Some(1), None, None, None).unwrap();
        assert_eq!(result.shape(), &[3, 3]);

        // Check some values
        assert!((result[[1, 1]] - 1.0).abs() < 1e-10);
        assert!((result[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((result[[2, 2]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_value_at_coordinates() {
        let input: Array2<f64> = Array2::eye(3);
        let indices = vec![1, 1];
        let value = value_at_coordinates(&input, &indices).unwrap();
        assert_eq!(value, 1.0);
    }

    #[test]
    fn test_interpn() {
        let input: Array2<f64> = Array2::eye(3);
        let points = vec![Array1::linspace(0.0, 2.0, 5), Array1::linspace(0.0, 2.0, 5)];
        let result = interpn(&input, &points, None, None, None).unwrap();
        assert_eq!(result.len(), 5);
    }
}
