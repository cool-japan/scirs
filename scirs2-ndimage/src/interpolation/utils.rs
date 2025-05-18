//! Utility functions for interpolation

// using these types only in function signatures
// use ndarray::Dimension;
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use super::BoundaryMode;
use crate::error::{NdimageError, Result};

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
pub fn handle_boundary<T>(coord: T, size: usize, mode: BoundaryMode) -> Result<T>
where
    T: Float + FromPrimitive + Debug,
{
    // Convert size to T for calculations
    let size_t = T::from_usize(size).unwrap();

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
pub fn linear_weights<T>(x: T) -> (usize, usize, T)
where
    T: Float + FromPrimitive + Debug,
{
    let x_floor = x.floor();
    let x_int = x_floor.to_usize().unwrap();
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
pub fn cubic_weights<T>(x: T) -> (usize, [T; 4])
where
    T: Float + FromPrimitive + Debug,
{
    let x_floor = x.floor();
    let x_int = x_floor.to_usize().unwrap();
    let t = x - x_floor;

    // Catmull-Rom cubic interpolation weights
    let t2 = t * t;
    let t3 = t2 * t;

    let half = T::from_f64(0.5).unwrap();
    let two = T::from_f64(2.0).unwrap();
    let three = T::from_f64(3.0).unwrap();
    let four = T::from_f64(4.0).unwrap();
    let five = T::from_f64(5.0).unwrap();

    let w0 = half * (-t3 + two * t2 - t);
    let w1 = half * (three * t3 - five * t2 + two);
    let w2 = half * (-three * t3 + four * t2 + t);
    let w3 = half * (t3 - t2);

    let weights = [w0, w1, w2, w3];

    // Starting index is one less than floor because cubic uses 4 points
    let start_idx = if x_int > 0 { x_int - 1 } else { 0 };

    (start_idx, weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handle_boundary_within_bounds() {
        let result = handle_boundary(1.5, 10, BoundaryMode::Nearest).unwrap();
        assert_eq!(result, 1.5);
    }

    #[test]
    fn test_handle_boundary_nearest() {
        let result = handle_boundary(-2.0, 10, BoundaryMode::Nearest).unwrap();
        assert_eq!(result, 0.0);

        let result = handle_boundary(15.0, 10, BoundaryMode::Nearest).unwrap();
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
