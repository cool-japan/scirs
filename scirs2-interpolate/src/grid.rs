//! Grid transformation and resampling utilities
//!
//! This module provides functions for transforming irregular data onto regular grids
//! and vice versa, as well as various grid manipulation utilities.

use crate::advanced::rbf::{RBFInterpolator, RBFKernel};
use crate::error::{InterpolateError, InterpolateResult};
use ndarray::{Array1, Array2, ArrayD, ArrayView1, ArrayView2, Axis};
use num_traits::{Float, FromPrimitive, Zero};
use std::fmt::Debug;

/// Grid transformation methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GridTransformMethod {
    /// Nearest neighbor assignment
    Nearest,
    /// Linear interpolation
    Linear,
    /// Cubic spline interpolation
    Cubic,
}

/// Create a grid of evenly spaced coordinates
///
/// # Arguments
///
/// * `bounds` - Min and max bounds for each dimension
/// * `shape` - Number of points in each dimension
///
/// # Returns
///
/// Vector of coordinate arrays, one for each dimension
///
/// # Examples
///
/// ```
/// use scirs2_interpolate::grid::create_regular_grid;
///
/// // Create a 10×20 grid from (0,0) to (1,2)
/// let grid_coords = create_regular_grid(
///     &[(0.0, 1.0), (0.0, 2.0)],
///     &[10, 20]
/// ).unwrap();
///
/// assert_eq!(grid_coords.len(), 2);
/// assert_eq!(grid_coords[0].len(), 10);
/// assert_eq!(grid_coords[1].len(), 20);
/// ```
#[allow(dead_code)]
pub fn create_regular_grid<F>(
    bounds: &[(F, F)],
    shape: &[usize],
) -> InterpolateResult<Vec<Array1<F>>>
where
    F: Float + FromPrimitive + Debug,
{
    if bounds.len() != shape.len() {
        return Err(InterpolateError::invalid_input(
            "bounds and shape must have the same length".to_string(),
        ));
    }

    let n_dims = bounds.len();
    let mut grid_coords = Vec::with_capacity(n_dims);

    for i in 0..n_dims {
        let (min, max) = bounds[i];
        let n_points = shape[i];

        if min >= max {
            return Err(InterpolateError::invalid_input(
                "min bound must be less than max bound".to_string(),
            ));
        }

        if n_points < 2 {
            return Err(InterpolateError::invalid_input(
                "grid shape must have at least 2 points in each dimension".to_string(),
            ));
        }

        let mut coords = Array1::zeros(n_points);

        if n_points == 1 {
            coords[0] = min;
        } else {
            let step = (max - min) / F::from_usize(n_points - 1).unwrap();
            for j in 0..n_points {
                coords[j] = min + F::from_usize(j).unwrap() * step;
            }
        }

        grid_coords.push(coords);
    }

    Ok(grid_coords)
}

/// Resample irregular (scattered) data onto a regular grid
///
/// # Arguments
///
/// * `points` - Coordinates of scattered data points (n_points × n_dimensions)
/// * `values` - Values at the scattered data points (n_points)
/// * `gridshape` - Shape of the output grid (number of points in each dimension)
/// * `grid_bounds` - Min and max bounds for each dimension of the grid
/// * `method` - Interpolation method to use
/// * `fill_value` - Value to use for grid points outside the convex hull of input points
///
/// # Returns
///
/// A tuple containing:
/// * The grid coordinates for each dimension (vector of arrays)
/// * The resampled values on the regular grid
#[allow(dead_code)]
pub fn resample_to_grid<F>(
    points: &ArrayView2<F>,
    values: &ArrayView1<F>,
    gridshape: &[usize],
    grid_bounds: &[(F, F)],
    method: GridTransformMethod,
    fill_value: F,
) -> InterpolateResult<(Vec<Array1<F>>, ArrayD<F>)>
where
    F: Float
        + FromPrimitive
        + Debug
        + Clone
        + PartialOrd
        + Zero
        + 'static
        + std::fmt::Display
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::fmt::LowerExp
        + Send
        + Sync,
{
    if points.nrows() != values.len() {
        return Err(InterpolateError::invalid_input(
            "Number of points and values must match".to_string(),
        ));
    }

    if points.ncols() != grid_bounds.len() {
        return Err(InterpolateError::invalid_input(
            "Point dimensions must match grid _bounds dimensions".to_string(),
        ));
    }

    if grid_bounds.len() != gridshape.len() {
        return Err(InterpolateError::invalid_input(
            "Grid _bounds and shape dimensions must match".to_string(),
        ));
    }

    // Create the regular grid
    let grid_coords = create_regular_grid(grid_bounds, gridshape)?;

    // Create a multidimensional array with the specified shape
    let shape: Vec<usize> = gridshape.to_vec();
    let mut grid_values = ArrayD::from_elem(shape.clone(), fill_value);

    match method {
        GridTransformMethod::Nearest => {
            resample_nearest_neighbor(points, values, &grid_coords, &mut grid_values, fill_value)?;
        }
        GridTransformMethod::Linear => {
            resample_linear(points, values, &grid_coords, &mut grid_values, fill_value)?;
        }
        GridTransformMethod::Cubic => {
            resample_rbf(points, values, &grid_coords, &mut grid_values, fill_value)?;
        }
    }

    Ok((grid_coords, grid_values))
}

/// Resample using nearest neighbor interpolation
#[allow(dead_code)]
fn resample_nearest_neighbor<F>(
    points: &ArrayView2<F>,
    values: &ArrayView1<F>,
    grid_coords: &[Array1<F>],
    grid_values: &mut ArrayD<F>,
    fill_value: F,
) -> InterpolateResult<()>
where
    F: Float + FromPrimitive + Debug + Clone + PartialOrd + Zero,
{
    let n_dims = grid_coords.len();

    // For each grid point, find the nearest data point
    let gridshape: Vec<usize> = grid_coords.iter().map(|coord| coord.len()).collect();

    // Generate all grid point coordinates
    let mut indices = vec![0; n_dims];

    loop {
        // Convert indices to actual coordinates
        let mut grid_point = vec![F::zero(); n_dims];
        for (dim, &idx) in indices.iter().enumerate() {
            grid_point[dim] = grid_coords[dim][idx];
        }

        // Find nearest data point
        let mut min_dist_sq = F::infinity();
        let mut nearest_value = fill_value;

        for i in 0..points.nrows() {
            let mut dist_sq = F::zero();
            for j in 0..n_dims {
                let diff = points[[i, j]] - grid_point[j];
                dist_sq = dist_sq + diff * diff;
            }

            if dist_sq < min_dist_sq {
                min_dist_sq = dist_sq;
                nearest_value = values[i];
            }
        }

        // Set the grid _value
        grid_values[&indices[..]] = nearest_value;

        // Increment indices
        if !increment_indices(&mut indices, &gridshape) {
            break;
        }
    }

    Ok(())
}

/// Resample using RBF interpolation for smooth results
#[allow(dead_code)]
fn resample_rbf<F>(
    points: &ArrayView2<F>,
    values: &ArrayView1<F>,
    grid_coords: &[Array1<F>],
    grid_values: &mut ArrayD<F>,
    fill_value: F,
) -> InterpolateResult<()>
where
    F: Float
        + FromPrimitive
        + Debug
        + Clone
        + PartialOrd
        + Zero
        + 'static
        + std::fmt::Display
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::fmt::LowerExp
        + Send
        + Sync,
{
    // Create RBF interpolator
    let rbf = RBFInterpolator::new(
        points,
        values,
        RBFKernel::Gaussian,
        F::from_f64(1.0).unwrap_or_else(|| F::one()),
    )?;

    let n_dims = grid_coords.len();
    let gridshape: Vec<usize> = grid_coords.iter().map(|coord| coord.len()).collect();
    let mut indices = vec![0; n_dims];

    loop {
        // Convert indices to actual coordinates
        let mut grid_point = Array1::zeros(n_dims);
        for (dim, &idx) in indices.iter().enumerate() {
            grid_point[dim] = grid_coords[dim][idx];
        }

        // Evaluate RBF at this grid point
        let interp_value = match rbf.interpolate(&grid_point.view().insert_axis(Axis(0))) {
            Ok(val) => val[0],
            Err(_) => fill_value,
        };

        grid_values[&indices[..]] = interp_value;

        // Increment indices
        if !increment_indices(&mut indices, &gridshape) {
            break;
        }
    }

    Ok(())
}

/// Linear interpolation for grid resampling (simplified implementation)
#[allow(dead_code)]
fn resample_linear<F>(
    points: &ArrayView2<F>,
    values: &ArrayView1<F>,
    grid_coords: &[Array1<F>],
    grid_values: &mut ArrayD<F>,
    fill_value: F,
) -> InterpolateResult<()>
where
    F: Float + FromPrimitive + Debug + Clone + PartialOrd + Zero,
{
    // For simplicity, fall back to nearest neighbor for multidimensional case
    // A full implementation would use multilinear interpolation
    resample_nearest_neighbor(points, values, grid_coords, grid_values, fill_value)
}

/// Helper function to increment multi-dimensional indices
#[allow(dead_code)]
fn increment_indices(indices: &mut [usize], shape: &[usize]) -> bool {
    for i in (0..indices.len()).rev() {
        indices[i] += 1;
        if indices[i] < shape[i] {
            return true;
        }
        indices[i] = 0;
    }
    false
}

/// Resample a regular grid to another regular grid with different resolution or bounds
///
/// # Arguments
///
/// * `src_coords` - Source grid coordinates (vector of arrays, one per dimension)
/// * `src_values` - Source grid values
/// * `dst_coords` - Destination grid coordinates (vector of arrays, one per dimension)
/// * `method` - Interpolation method to use
/// * `fill_value` - Value to use for grid points outside the source grid
///
/// # Returns
///
/// Resampled values on the destination grid
#[allow(dead_code)]
pub fn resample_grid_to_grid<F, D>(
    src_coords: &[Array1<F>],
    src_values: &ndarray::ArrayView<F, D>,
    dst_coords: &[Array1<F>],
    method: GridTransformMethod,
    fill_value: F,
) -> InterpolateResult<ArrayD<F>>
where
    F: Float + FromPrimitive + Debug + Clone + PartialOrd + Zero + 'static,
    D: ndarray::Dimension,
{
    if src_coords.len() != dst_coords.len() {
        return Err(InterpolateError::invalid_input(
            "Source and destination must have same number of dimensions".to_string(),
        ));
    }

    let _n_dims = src_coords.len(); // Reserved for future use

    // Verify source coordinates match source _values shape
    for (i, coord) in src_coords.iter().enumerate() {
        if coord.len() != src_values.shape()[i] {
            return Err(InterpolateError::invalid_input(format!(
                "Source coordinate dimension {} length doesn't match _values shape",
                i
            )));
        }
    }

    // Create destination grid shape
    let dstshape: Vec<usize> = dst_coords.iter().map(|coord| coord.len()).collect();
    let mut dst_values = ArrayD::from_elem(dstshape.clone(), fill_value);

    match method {
        GridTransformMethod::Nearest => {
            grid_to_grid_nearest(
                src_coords,
                src_values,
                dst_coords,
                &mut dst_values,
                fill_value,
            )?;
        }
        GridTransformMethod::Linear => {
            grid_to_grid_linear(
                src_coords,
                src_values,
                dst_coords,
                &mut dst_values,
                fill_value,
            )?;
        }
        GridTransformMethod::Cubic => {
            // For cubic, we'll use linear interpolation as it's more stable for grids
            grid_to_grid_linear(
                src_coords,
                src_values,
                dst_coords,
                &mut dst_values,
                fill_value,
            )?;
        }
    }

    Ok(dst_values)
}

/// Convert multi-dimensional indices to linear index
#[allow(dead_code)]
fn ravel_multi_index(indices: &[usize], shape: &[usize]) -> usize {
    let mut linear_idx = 0;
    let mut stride = 1;

    for i in (0..indices.len()).rev() {
        linear_idx += indices[i] * stride;
        stride *= shape[i];
    }

    linear_idx
}

/// Grid-to-grid resampling using nearest neighbor
#[allow(dead_code)]
fn grid_to_grid_nearest<F, D>(
    src_coords: &[Array1<F>],
    src_values: &ndarray::ArrayView<F, D>,
    dst_coords: &[Array1<F>],
    dst_values: &mut ArrayD<F>,
    fill_value: F,
) -> InterpolateResult<()>
where
    F: Float + FromPrimitive + Debug + Clone + PartialOrd + Zero,
    D: ndarray::Dimension,
{
    let n_dims = src_coords.len();
    let dstshape: Vec<usize> = dst_coords.iter().map(|coord| coord.len()).collect();
    let mut indices = vec![0; n_dims];

    loop {
        // Get destination grid point coordinates
        let mut dst_point = vec![F::zero(); n_dims];
        for (dim, &idx) in indices.iter().enumerate() {
            dst_point[dim] = dst_coords[dim][idx];
        }

        // Find nearest source grid indices
        let mut src_indices = vec![0; n_dims];
        let valid = true;

        for dim in 0..n_dims {
            let coord = dst_point[dim];
            let src_coord = &src_coords[dim];

            // Find nearest index in source coordinates
            let mut best_idx = 0;
            let mut min_dist = (src_coord[0] - coord).abs();

            for (i, &src_val) in src_coord.iter().enumerate() {
                let dist = (src_val - coord).abs();
                if dist < min_dist {
                    min_dist = dist;
                    best_idx = i;
                }
            }

            src_indices[dim] = best_idx;
        }

        // Get the source _value and assign to destination
        if valid {
            // Use linear indexing for all cases to avoid generic dimension issues
            let linear_idx = ravel_multi_index(&src_indices, &src_values.shape());
            let src_value = src_values.as_slice().unwrap()[linear_idx];
            let dst_linear_idx = ravel_multi_index(&indices, &dstshape);
            dst_values.as_slice_mut().unwrap()[dst_linear_idx] = src_value;
        } else {
            let dst_linear_idx = ravel_multi_index(&indices, &dstshape);
            dst_values.as_slice_mut().unwrap()[dst_linear_idx] = fill_value;
        }

        // Increment indices
        if !increment_indices(&mut indices, &dstshape) {
            break;
        }
    }

    Ok(())
}

/// Grid-to-grid resampling using linear interpolation
#[allow(dead_code)]
fn grid_to_grid_linear<F, D>(
    src_coords: &[Array1<F>],
    src_values: &ndarray::ArrayView<F, D>,
    dst_coords: &[Array1<F>],
    dst_values: &mut ArrayD<F>,
    fill_value: F,
) -> InterpolateResult<()>
where
    F: Float + FromPrimitive + Debug + Clone + PartialOrd + Zero,
    D: ndarray::Dimension,
{
    let n_dims = src_coords.len();
    let dstshape: Vec<usize> = dst_coords.iter().map(|coord| coord.len()).collect();
    let mut indices = vec![0; n_dims];

    loop {
        // Get destination grid point coordinates
        let mut dst_point = vec![F::zero(); n_dims];
        for (dim, &idx) in indices.iter().enumerate() {
            dst_point[dim] = dst_coords[dim][idx];
        }

        // Perform multilinear interpolation
        let interpolated_value =
            multilinear_interpolate(src_coords, src_values, &dst_point, fill_value)?;

        dst_values[&indices[..]] = interpolated_value;

        // Increment indices
        if !increment_indices(&mut indices, &dstshape) {
            break;
        }
    }

    Ok(())
}

/// Perform multilinear interpolation at a single point
#[allow(dead_code)]
fn multilinear_interpolate<F, D>(
    coords: &[Array1<F>],
    values: &ndarray::ArrayView<F, D>,
    point: &[F],
    fill_value: F,
) -> InterpolateResult<F>
where
    F: Float + FromPrimitive + Debug + Clone + PartialOrd + Zero,
    D: ndarray::Dimension,
{
    let n_dims = coords.len();

    // Find bounding grid cells for each dimension
    let mut lower_indices = vec![0; n_dims];
    let mut upper_indices = vec![0; n_dims];
    let mut weights = vec![F::zero(); n_dims];

    for dim in 0..n_dims {
        let coord_array = &coords[dim];
        let target = point[dim];

        // Find the interval containing the target
        let mut found = false;
        for i in 0..coord_array.len() - 1 {
            if target >= coord_array[i] && target <= coord_array[i + 1] {
                lower_indices[dim] = i;
                upper_indices[dim] = i + 1;

                // Calculate interpolation weight
                let dx = coord_array[i + 1] - coord_array[i];
                if dx.abs() > F::zero() {
                    weights[dim] = (target - coord_array[i]) / dx;
                } else {
                    weights[dim] = F::zero();
                }
                found = true;
                break;
            }
        }

        if !found {
            // Point is outside grid bounds
            return Ok(fill_value);
        }
    }

    // Perform multilinear interpolation
    // For N dimensions, we need 2^N corner values
    let n_corners = 1 << n_dims; // 2^n_dims
    let mut result = F::zero();

    for corner in 0..n_corners {
        let mut corner_indices = vec![0; n_dims];
        let mut corner_weight = F::one();

        for dim in 0..n_dims {
            if (corner >> dim) & 1 == 0 {
                corner_indices[dim] = lower_indices[dim];
                corner_weight = corner_weight * (F::one() - weights[dim]);
            } else {
                corner_indices[dim] = upper_indices[dim];
                corner_weight = corner_weight * weights[dim];
            }
        }

        // Get _value at this corner using linear indexing
        let linear_idx = ravel_multi_index(&corner_indices, &values.shape());
        let corner_value = values.as_slice().unwrap()[linear_idx];
        result = result + corner_weight * corner_value;
    }

    Ok(result)
}

/// Maps values from a regular grid to arbitrary points using interpolation
///
/// # Arguments
///
/// * `grid_coords` - Grid coordinates (vector of arrays, one per dimension)
/// * `grid_values` - Values on the regular grid
/// * `query_points` - Points at which to evaluate (n_points × n_dimensions)
/// * `method` - Interpolation method to use
/// * `fill_value` - Value to use for query points outside the grid
///
/// # Returns
///
/// Values at the query points
#[allow(dead_code)]
pub fn map_grid_to_points<F, D>(
    grid_coords: &[Array1<F>],
    grid_values: &ndarray::ArrayView<F, D>,
    query_points: &ArrayView2<F>,
    method: GridTransformMethod,
    fill_value: F,
) -> InterpolateResult<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Clone + PartialOrd + Zero + 'static,
    D: ndarray::Dimension,
{
    let n_points = query_points.nrows();
    let n_dims = query_points.ncols();

    if grid_coords.len() != n_dims {
        return Err(InterpolateError::invalid_input(
            "Grid coordinates and query point dimensions must match".to_string(),
        ));
    }

    // Verify grid coordinates match grid _values shape
    for (i, coord) in grid_coords.iter().enumerate() {
        if coord.len() != grid_values.shape()[i] {
            return Err(InterpolateError::invalid_input(format!(
                "Grid coordinate dimension {} length doesn't match _values shape",
                i
            )));
        }
    }

    let mut result = Array1::zeros(n_points);

    for i in 0..n_points {
        let query_point: Vec<F> = query_points.row(i).to_vec();

        let interpolated_value = match method {
            GridTransformMethod::Nearest => {
                grid_nearest_neighbor(grid_coords, grid_values, &query_point, fill_value)?
            }
            GridTransformMethod::Linear => {
                multilinear_interpolate(grid_coords, grid_values, &query_point, fill_value)?
            }
            GridTransformMethod::Cubic => {
                // For cubic, we fall back to linear for stability
                multilinear_interpolate(grid_coords, grid_values, &query_point, fill_value)?
            }
        };

        result[i] = interpolated_value;
    }

    Ok(result)
}

/// Find the nearest grid point value
#[allow(dead_code)]
fn grid_nearest_neighbor<F, D>(
    grid_coords: &[Array1<F>],
    grid_values: &ndarray::ArrayView<F, D>,
    query_point: &[F],
    _fill_value: F,
) -> InterpolateResult<F>
where
    F: Float + FromPrimitive + Debug + Clone + PartialOrd + Zero,
    D: ndarray::Dimension,
{
    let n_dims = grid_coords.len();
    let mut nearest_indices = vec![0; n_dims];

    for dim in 0..n_dims {
        let coord_array = &grid_coords[dim];
        let target = query_point[dim];

        // Find nearest index
        let mut best_idx = 0;
        let mut min_dist = (coord_array[0] - target).abs();

        for (i, &coord_val) in coord_array.iter().enumerate() {
            let dist = (coord_val - target).abs();
            if dist < min_dist {
                min_dist = dist;
                best_idx = i;
            }
        }

        nearest_indices[dim] = best_idx;
    }

    let linear_idx = ravel_multi_index(&nearest_indices, &grid_values.shape());
    Ok(grid_values.as_slice().unwrap()[linear_idx])
}

/// Efficient grid coordinate range checking
#[allow(dead_code)]
fn point_in_grid_bounds<F>(_gridcoords: &[Array1<F>], point: &[F]) -> bool
where
    F: Float + PartialOrd,
{
    for (dim, coord_array) in _gridcoords.iter().enumerate() {
        let target = point[dim];
        let min_coord = coord_array[0];
        let max_coord = coord_array[coord_array.len() - 1];

        if target < min_coord || target > max_coord {
            return false;
        }
    }
    true
}

/// Create a tensor product grid from coordinate arrays
///
/// This function creates all combinations of coordinates from the input arrays,
/// useful for creating meshgrids for evaluation.
#[allow(dead_code)]
pub fn create_meshgrid<F>(coords: &[Array1<F>]) -> InterpolateResult<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Clone + Zero,
{
    let n_dims = coords.len();
    if n_dims == 0 {
        return Err(InterpolateError::invalid_input(
            "At least one coordinate array required".to_string(),
        ));
    }

    // Calculate total number of grid points
    let mut total_points = 1;
    for coord in coords {
        total_points *= coord.len();
    }

    let mut result = Array2::zeros((total_points, n_dims));
    let shapes: Vec<usize> = coords.iter().map(|c| c.len()).collect();
    let mut indices = vec![0; n_dims];

    for row in 0..total_points {
        // Set coordinates for this grid point
        for (dim, &idx) in indices.iter().enumerate() {
            result[[row, dim]] = coords[dim][idx];
        }

        // Increment multi-dimensional indices
        increment_indices(&mut indices, &shapes);
    }

    Ok(result)
}

/// Calculate grid spacing for each dimension
#[allow(dead_code)]
pub fn calculate_grid_spacing<F>(coords: &[Array1<F>]) -> InterpolateResult<Vec<F>>
where
    F: Float + FromPrimitive + Debug + Clone,
{
    let mut spacings = Vec::with_capacity(coords.len());

    for coord in coords {
        if coord.len() < 2 {
            return Err(InterpolateError::invalid_input(
                "Grid coordinates must have at least 2 points".to_string(),
            ));
        }

        // Calculate average spacing (assumes roughly uniform grid)
        let total_range = coord[coord.len() - 1] - coord[0];
        let n_intervals = F::from_usize(coord.len() - 1).unwrap();
        let avg_spacing = total_range / n_intervals;

        spacings.push(avg_spacing);
    }

    Ok(spacings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    // テスト関数で使用するモジュール

    #[test]
    fn test_create_regular_grid() {
        // 1D grid
        let grid_1d = create_regular_grid(&[(0.0, 1.0)], &[5]).unwrap();

        assert_eq!(grid_1d.len(), 1);
        assert_eq!(grid_1d[0].len(), 5);
        assert_abs_diff_eq!(grid_1d[0][0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(grid_1d[0][4], 1.0, epsilon = 1e-10);

        // 2D grid
        let grid_2d = create_regular_grid(&[(0.0, 1.0), (-1.0, 1.0)], &[3, 5]).unwrap();

        assert_eq!(grid_2d.len(), 2);
        assert_eq!(grid_2d[0].len(), 3);
        assert_eq!(grid_2d[1].len(), 5);
        assert_abs_diff_eq!(grid_2d[0][0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(grid_2d[0][2], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(grid_2d[1][0], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(grid_2d[1][4], 1.0, epsilon = 1e-10);
    }
}
