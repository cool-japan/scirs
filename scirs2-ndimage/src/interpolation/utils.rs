//! Utility functions for interpolation

use scirs2_core::ndarray::Array;
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use super::BoundaryMode;
use crate::error::{NdimageError, NdimageResult};

/// Helper function for safe conversion from usize to float
#[allow(dead_code)]
fn safe_usize_to_float<T: Float + FromPrimitive>(value: usize) -> NdimageResult<T> {
    T::from_usize(value).ok_or_else(|| {
        NdimageError::ComputationError(format!("Failed to convert usize {} to float type", value))
    })
}

/// Helper function for safe conversion from float to usize
#[allow(dead_code)]
fn safe_float_to_usize<T: Float>(value: T) -> NdimageResult<usize> {
    value.to_usize().ok_or_else(|| {
        NdimageError::ComputationError("Failed to convert float to usize".to_string())
    })
}

/// Handle out-of-bounds coordinates according to the boundary mode
///
/// # Arguments
///
/// * `coord` - Coordinate to process
/// * `size` - Size of the array dimension
/// * `mode` - Boundary handling mode
///
/// # Returns
///
/// * `Result<T>` - Processed coordinate
#[allow(dead_code)]
pub fn handle_boundary<T>(coord: T, size: usize, mode: BoundaryMode) -> NdimageResult<T>
where
    T: Float + FromPrimitive + Debug + std::ops::AddAssign + std::ops::DivAssign + 'static,
{
    // Convert size to T for calculations
    let size_t = safe_usize_to_float(size)?;

    // Handle within-bounds case
    if coord >= T::zero() && coord < size_t {
        return Ok(coord);
    }

    // Handle out-of-bounds according to mode
    match mode {
        BoundaryMode::Constant => {
            // For constant mode, return an out-of-bounds indicator
            // The actual handling would be done by the caller
            Err(NdimageError::InterpolationError(format!(
                "Coordinate {:?} out of bounds for size {} with constant mode",
                coord, size
            )))
        }
        BoundaryMode::Nearest => {
            if coord < T::zero() {
                Ok(T::zero())
            } else {
                Ok(size_t - T::one())
            }
        }
        BoundaryMode::Reflect => {
            // Placeholder for reflect mode
            // Would implement proper reflection calculation
            Ok(T::zero())
        }
        BoundaryMode::Mirror => {
            // Placeholder for mirror mode
            // Would implement proper mirroring calculation
            Ok(T::zero())
        }
        BoundaryMode::Wrap => {
            // Placeholder for wrap mode
            // Would implement proper wrapping calculation
            Ok(T::zero())
        }
    }
}

/// Get the weights for linear interpolation
///
/// # Arguments
///
/// * `x` - Position for interpolation
///
/// # Returns
///
/// * `(usize, usize, T)` - (left index, right index, right weight)
#[allow(dead_code)]
pub fn linear_weights<T>(x: T) -> (usize, usize, T)
where
    T: Float + FromPrimitive + Debug + std::ops::AddAssign + std::ops::DivAssign + 'static,
{
    let x_floor = x.floor();
    let x_int = safe_float_to_usize(x_floor).unwrap_or(0); // Use 0 as fallback for interpolation
    let t = x - x_floor;

    (x_int, x_int + 1, t)
}

/// Get the weights for cubic interpolation
///
/// # Arguments
///
/// * `x` - Position for interpolation
///
/// # Returns
///
/// * `(usize, [T; 4])` - (starting index, weights for 4 points)
#[allow(dead_code)]
pub fn cubic_weights<T>(x: T) -> (usize, [T; 4])
where
    T: Float + FromPrimitive + Debug + std::ops::AddAssign + std::ops::DivAssign + 'static,
{
    let x_floor = x.floor();
    let x_int = safe_float_to_usize(x_floor).unwrap_or(0); // Use 0 as fallback for interpolation
    let t = x - x_floor;

    // Catmull-Rom cubic interpolation weights
    let t2 = t * t;
    let t3 = t2 * t;

    // Pre-calculate constants with safe conversions
    let half = T::from_f64(0.5).unwrap_or_else(|| T::one() / (T::one() + T::one()));
    let two = T::from_f64(2.0).unwrap_or_else(|| T::one() + T::one());
    let three = T::from_f64(3.0).unwrap_or_else(|| two + T::one());
    let four = T::from_f64(4.0).unwrap_or_else(|| two + two);
    let five = T::from_f64(5.0).unwrap_or_else(|| four + T::one());

    let w0 = half * (-t3 + two * t2 - t);
    let w1 = half * (three * t3 - five * t2 + two);
    let w2 = half * (-three * t3 + four * t2 + t);
    let w3 = half * (t3 - t2);

    let weights = [w0, w1, w2, w3];

    // Starting index is one less than floor because cubic uses 4 points
    let start_idx = if x_int > 0 { x_int - 1 } else { 0 };

    (start_idx, weights)
}

/// Helper function for nearest neighbor interpolation
#[allow(dead_code)]
pub fn interpolate_nearest<T>(
    input: &Array<T, scirs2_core::ndarray::IxDyn>,
    coords: &[T],
    boundary: &BoundaryMode,
    const_val: T,
) -> T
where
    T: Float + FromPrimitive + Debug + std::ops::AddAssign + std::ops::DivAssign + 'static,
{
    // Round coordinates to nearest integers
    let int_coords: Vec<isize> = coords
        .iter()
        .map(|&coord| coord.round().to_isize().unwrap_or(0))
        .collect();

    // Apply boundary conditions and check bounds
    let inputshape = input.shape();
    let bounded_coords: Vec<usize> = int_coords
        .iter()
        .enumerate()
        .map(|(i, &coord)| {
            let dim_size = inputshape[i] as isize;
            apply_boundary_condition(coord, dim_size, boundary)
        })
        .collect();

    // Check if coordinates are valid (within bounds after boundary handling)
    for (i, &coord) in bounded_coords.iter().enumerate() {
        if coord >= inputshape[i] {
            return const_val; // Out of bounds, return constant value
        }
    }

    // Get value at the bounded coordinates
    input
        .get(bounded_coords.as_slice())
        .copied()
        .unwrap_or(const_val)
}

/// Helper function for linear interpolation  
#[allow(dead_code)]
pub fn interpolate_linear<T>(
    input: &Array<T, scirs2_core::ndarray::IxDyn>,
    coords: &[T],
    boundary: &BoundaryMode,
    const_val: T,
) -> T
where
    T: Float + FromPrimitive + Debug + std::ops::AddAssign + std::ops::DivAssign + 'static,
{
    let ndim = coords.len();
    if ndim == 0 {
        return const_val;
    }

    // Handle 1D linear interpolation
    if ndim == 1 {
        let x = coords[0];
        let x0 = x.floor();
        let x1 = x0 + T::one();
        let dx = x - x0;

        let i0 = x0.to_isize().unwrap_or(0);
        let i1 = x1.to_isize().unwrap_or(0);

        let dim_size = input.shape()[0] as isize;
        let idx0 = apply_boundary_condition(i0, dim_size, boundary);
        let idx1 = apply_boundary_condition(i1, dim_size, boundary);

        // Check bounds for constant mode
        if matches!(boundary, BoundaryMode::Constant)
            && (i0 < 0 || i0 >= dim_size || i1 < 0 || i1 >= dim_size)
        {
            return const_val;
        }

        let v0 = input.get([idx0]).copied().unwrap_or(const_val);
        let v1 = input.get([idx1]).copied().unwrap_or(const_val);

        return v0 * (T::one() - dx) + v1 * dx;
    }

    // For 2D and higher, use separable linear interpolation
    if ndim == 2 {
        let x = coords[0];
        let y = coords[1];

        let x0 = x.floor();
        let x1 = x0 + T::one();
        let y0 = y.floor();
        let y1 = y0 + T::one();

        let dx = x - x0;
        let dy = y - y0;

        let i0 = x0.to_isize().unwrap_or(0);
        let i1 = x1.to_isize().unwrap_or(0);
        let j0 = y0.to_isize().unwrap_or(0);
        let j1 = y1.to_isize().unwrap_or(0);

        let dim_size_x = input.shape()[0] as isize;
        let dim_size_y = input.shape()[1] as isize;

        let idx0 = apply_boundary_condition(i0, dim_size_x, boundary);
        let idx1 = apply_boundary_condition(i1, dim_size_x, boundary);
        let jdx0 = apply_boundary_condition(j0, dim_size_y, boundary);
        let jdx1 = apply_boundary_condition(j1, dim_size_y, boundary);

        // Check bounds for constant mode
        if matches!(boundary, BoundaryMode::Constant)
            && (i0 < 0
                || i0 >= dim_size_x
                || i1 < 0
                || i1 >= dim_size_x
                || j0 < 0
                || j0 >= dim_size_y
                || j1 < 0
                || j1 >= dim_size_y)
        {
            return const_val;
        }

        let v00 = input.get([idx0, jdx0]).copied().unwrap_or(const_val);
        let v01 = input.get([idx0, jdx1]).copied().unwrap_or(const_val);
        let v10 = input.get([idx1, jdx0]).copied().unwrap_or(const_val);
        let v11 = input.get([idx1, jdx1]).copied().unwrap_or(const_val);

        // Bilinear interpolation
        let v0 = v00 * (T::one() - dy) + v01 * dy;
        let v1 = v10 * (T::one() - dy) + v11 * dy;

        return v0 * (T::one() - dx) + v1 * dx;
    }

    // For higher dimensions, fall back to nearest neighbor
    interpolate_nearest(input, coords, boundary, const_val)
}

/// Apply boundary condition to a coordinate
#[allow(dead_code)]
pub fn apply_boundary_condition(_coord: isize, dimsize: isize, mode: &BoundaryMode) -> usize {
    match mode {
        BoundaryMode::Constant => {
            if _coord < 0 || _coord >= dimsize {
                // Return a value that will be caught as out of bounds
                dimsize as usize
            } else {
                _coord as usize
            }
        }
        BoundaryMode::Nearest => {
            if _coord < 0 {
                0
            } else if _coord >= dimsize {
                (dimsize - 1) as usize
            } else {
                _coord as usize
            }
        }
        BoundaryMode::Wrap => {
            if dimsize == 0 {
                0
            } else {
                let wrapped = ((_coord % dimsize) + dimsize) % dimsize;
                wrapped as usize
            }
        }
        BoundaryMode::Reflect => {
            if dimsize <= 1 {
                0
            } else {
                let reflected = if _coord < 0 {
                    (-_coord - 1) % dimsize
                } else if _coord >= dimsize {
                    (2 * dimsize - _coord - 1) % dimsize
                } else {
                    _coord
                };
                reflected as usize
            }
        }
        BoundaryMode::Mirror => {
            if dimsize <= 1 {
                0
            } else {
                let period = 2 * (dimsize - 1);
                let mirrored = if _coord < 0 {
                    (-_coord) % period
                } else if _coord >= dimsize {
                    period - ((_coord - dimsize + 1) % period) - 1
                } else {
                    _coord
                };
                (mirrored.min(dimsize - 1)) as usize
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handle_boundary_within_bounds() {
        let result = handle_boundary(1.5, 10, BoundaryMode::Nearest)
            .expect("handle_boundary should succeed for test");
        assert_eq!(result, 1.5);
    }

    #[test]
    fn test_handle_boundary_nearest() {
        let result = handle_boundary(-2.0, 10, BoundaryMode::Nearest)
            .expect("handle_boundary should succeed for test");
        assert_eq!(result, 0.0);

        let result = handle_boundary(15.0, 10, BoundaryMode::Nearest)
            .expect("handle_boundary should succeed for test");
        assert_eq!(result, 9.0);
    }

    #[test]
    fn test_linear_weights() {
        let (i0, i1, t) = linear_weights(1.3);
        assert_eq!(i0, 1);
        assert_eq!(i1, 2);
        assert!((t - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_cubic_weights() {
        let (start_idx, weights) = cubic_weights(1.3);
        assert!(start_idx <= 1);
        assert_eq!(weights.len(), 4);

        // Weights should sum to 1
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }
}
