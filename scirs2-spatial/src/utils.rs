//! Utility functions for spatial algorithms
//!
//! This module provides utility functions for spatial algorithms.

use crate::error::{SpatialError, SpatialResult};
use scirs2_core::ndarray::Array2;

/// Type alias for the result of scaling points: (scaled points, scaling factors)
type ScaledPointsResult = SpatialResult<(Array2<f64>, Vec<(f64, f64)>)>;

/// Check if two points are equal within a tolerance
///
/// # Arguments
///
/// * `point1` - First point
/// * `point2` - Second point
/// * `tol` - Tolerance (default: 1e-8)
///
/// # Returns
///
/// * True if points are equal within tolerance
#[allow(dead_code)]
pub fn points_equal<T>(point1: &[T], point2: &[T], tol: Option<T>) -> bool
where
    T: PartialOrd + std::ops::Sub<Output = T> + Copy + scirs2_core::numeric::FromPrimitive,
{
    // Default tolerance as 1e-8 converted to type T
    let tol = match tol {
        Some(t) => t,
        None => match T::from_f64(1e-8) {
            Some(t) => t,
            None => {
                // Fallback: use a very small value that should work for most numeric types
                // This handles edge cases where 1e-8 cannot be represented in type T
                match T::from_f64(0.0) {
                    Some(zero) => zero,   // Use zero as fallback tolerance (exact equality)
                    None => return false, // If we can't even create zero, points can't be equal
                }
            }
        },
    };

    if point1.len() != point2.len() {
        return false;
    }

    for i in 0..point1.len() {
        if point1[i] > point2[i] && point1[i] - point2[i] > tol {
            return false;
        }
        if point2[i] > point1[i] && point2[i] - point1[i] > tol {
            return false;
        }
    }

    true
}

/// Scale points to the range [0, 1] in each dimension
///
/// # Arguments
///
/// * `points` - Array of points to scale
///
/// # Returns
///
/// * Scaled points and scale factors (min, range)
#[allow(dead_code)]
pub fn scale_points(points: &Array2<f64>) -> ScaledPointsResult {
    let n = points.nrows();
    let d = points.ncols();

    if n == 0 {
        return Err(SpatialError::ValueError("Empty point set".to_string()));
    }

    // Find min and max for each dimension
    let mut mins = vec![f64::INFINITY; d];
    let mut maxs = vec![f64::NEG_INFINITY; d];

    for i in 0..n {
        for j in 0..d {
            let val = points[[i, j]];
            mins[j] = mins[j].min(val);
            maxs[j] = maxs[j].max(val);
        }
    }

    // Compute ranges
    let mut ranges = vec![];
    for i in 0..d {
        ranges.push((mins[i], maxs[i] - mins[i]));
    }

    // Scale _points
    let mut scaled = Array2::zeros((n, d));
    for i in 0..n {
        for j in 0..d {
            if ranges[j].1 > 0.0 {
                scaled[[i, j]] = (points[[i, j]] - ranges[j].0) / ranges[j].1;
            } else {
                scaled[[i, j]] = 0.5; // All _points have same value in this dimension
            }
        }
    }

    Ok((scaled, ranges))
}

/// Unscale points from the range [0, 1] to original range
///
/// # Arguments
///
/// * `points` - Scaled points
/// * `ranges` - Scale factors (min, range) for each dimension
///
/// # Returns
///
/// * Unscaled points
#[allow(dead_code)]
pub fn unscale_points(points: &Array2<f64>, ranges: &[(f64, f64)]) -> SpatialResult<Array2<f64>> {
    let n = points.nrows();
    let d = points.ncols();

    if d != ranges.len() {
        return Err(SpatialError::DimensionError(format!(
            "Points dimension ({}) does not match ranges dimension ({})",
            d,
            ranges.len()
        )));
    }

    let mut unscaled = Array2::zeros((n, d));
    for i in 0..n {
        for j in 0..d {
            unscaled[[i, j]] = points[[i, j]] * ranges[j].1 + ranges[j].0;
        }
    }

    Ok(unscaled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::arr2;

    #[test]
    fn test_points_equal() {
        let point1 = [1.0, 2.0, 3.0];
        let point2 = [1.0, 2.0, 3.0];
        let point3 = [1.0, 2.0, 3.00001];
        let point4 = [1.0, 2.0, 3.1];

        assert!(points_equal(&point1, &point2, None));
        // Within epsilon 1e-8 (which is the default tolerance)
        // Note: In a real test we'd use within 1e-4 for floating point comparisons
        let tolerance = 1e-4; // Larger tolerance for test
        assert!(points_equal(&point1, &point3, Some(tolerance)));
        assert!(!points_equal(&point1, &point4, None)); // Outside default tolerance

        // With custom tolerance
        assert!(points_equal(&point1, &point3, Some(0.001)));
        assert!(!points_equal(&point1, &point3, Some(0.000001)));
    }

    #[test]
    fn test_scale_unscale_points() {
        let points = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);

        let (scaled, ranges) = scale_points(&points).unwrap();

        // Check that scaled points are in [0, 1]
        for i in 0..scaled.nrows() {
            for j in 0..scaled.ncols() {
                assert!(scaled[[i, j]] >= 0.0 && scaled[[i, j]] <= 1.0);
            }
        }

        // Check ranges
        assert_eq!(ranges[0], (1.0, 6.0)); // x: min=1, range=6
        assert_eq!(ranges[1], (2.0, 6.0)); // y: min=2, range=6
        assert_eq!(ranges[2], (3.0, 6.0)); // z: min=3, range=6

        // Unscale and check
        let unscaled = unscale_points(&scaled, &ranges).unwrap();

        for i in 0..points.nrows() {
            for j in 0..points.ncols() {
                assert_relative_eq!(points[[i, j]], unscaled[[i, j]], epsilon = 1e-10);
            }
        }
    }
}
