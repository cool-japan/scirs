//! Volume computation for convex hulls
//!
//! This module provides volume (hypervolume) calculations for convex hulls
//! in various dimensions, using appropriate methods for each dimensionality.

use crate::convex_hull::core::ConvexHull;
use crate::convex_hull::geometry::{
    compute_high_dim_volume, compute_polygon_area, compute_polyhedron_volume,
};
use crate::error::SpatialResult;

/// Compute the volume of a convex hull
///
/// The volume computation method depends on the dimensionality:
/// - 1D: Length (distance between extreme points)
/// - 2D: Area using shoelace formula
/// - 3D: Volume using tetrahedron decomposition
/// - nD: High-dimensional volume approximation using facet equations
///
/// # Arguments
///
/// * `hull` - The convex hull to compute volume for
///
/// # Returns
///
/// * Result containing the volume/area/length of the convex hull
///
/// # Examples
///
/// ```rust
/// use scirs2_spatial::convex_hull::ConvexHull;
/// use scirs2_spatial::convex_hull::properties::volume::compute_volume;
/// use ndarray::array;
///
/// // 2D square with area 1
/// let points = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
/// let hull = ConvexHull::new(&points.view()).unwrap();
/// let area = compute_volume(&hull).unwrap();
/// assert!((area - 1.0).abs() < 1e-10);
/// ```
pub fn compute_volume(hull: &ConvexHull) -> SpatialResult<f64> {
    match hull.ndim() {
        1 => compute_1d_volume(hull),
        2 => compute_2d_volume(hull),
        3 => compute_3d_volume(hull),
        _ => compute_nd_volume(hull),
    }
}

/// Compute 1D volume (length of line segment)
///
/// # Arguments
///
/// * `hull` - The convex hull (should be 1D)
///
/// # Returns
///
/// * Result containing the length of the 1D hull
fn compute_1d_volume(hull: &ConvexHull) -> SpatialResult<f64> {
    if hull.vertex_indices.len() < 2 {
        return Ok(0.0);
    }

    let min_idx = *hull.vertex_indices.iter().min().unwrap();
    let max_idx = *hull.vertex_indices.iter().max().unwrap();

    Ok((hull.points[[max_idx, 0]] - hull.points[[min_idx, 0]]).abs())
}

/// Compute 2D volume (area using shoelace formula)
///
/// # Arguments
///
/// * `hull` - The convex hull (should be 2D)
///
/// # Returns
///
/// * Result containing the area of the 2D hull
fn compute_2d_volume(hull: &ConvexHull) -> SpatialResult<f64> {
    if hull.vertex_indices.len() < 3 {
        return Ok(0.0);
    }

    compute_polygon_area(&hull.points.view(), &hull.vertex_indices)
}

/// Compute 3D volume using tetrahedron decomposition
///
/// # Arguments
///
/// * `hull` - The convex hull (should be 3D)
///
/// # Returns
///
/// * Result containing the volume of the 3D hull
fn compute_3d_volume(hull: &ConvexHull) -> SpatialResult<f64> {
    if hull.vertex_indices.len() < 4 {
        return Ok(0.0);
    }

    compute_polyhedron_volume(&hull.points.view(), &hull.simplices)
}

/// Compute high-dimensional volume using facet equations
///
/// # Arguments
///
/// * `hull` - The convex hull (should be nD where n > 3)
///
/// # Returns
///
/// * Result containing the high-dimensional volume
fn compute_nd_volume(hull: &ConvexHull) -> SpatialResult<f64> {
    if let Some(equations) = &hull.equations {
        compute_high_dim_volume(&hull.points.view(), &hull.vertex_indices, &equations.view())
    } else {
        use crate::error::SpatialError;
        Err(SpatialError::NotImplementedError(
            "Volume computation for dimensions > 3 requires facet equations".to_string(),
        ))
    }
}

/// Compute volume using Monte Carlo estimation
///
/// This is an alternative method for high-dimensional volume computation
/// that can be useful when facet equations are not available.
///
/// # Arguments
///
/// * `hull` - The convex hull
/// * `num_samples` - Number of Monte Carlo samples to use
///
/// # Returns
///
/// * Result containing the estimated volume
///
/// # Examples
///
/// ```rust
/// use scirs2_spatial::convex_hull::ConvexHull;
/// use scirs2_spatial::convex_hull::properties::volume::compute_volume_monte_carlo;
/// use ndarray::array;
///
/// let points = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
/// let hull = ConvexHull::new(&points.view()).unwrap();
/// let estimated_area = compute_volume_monte_carlo(&hull, 10000).unwrap();
/// assert!((estimated_area - 1.0).abs() < 0.1); // Monte Carlo approximation
/// ```
pub fn compute_volume_monte_carlo(hull: &ConvexHull, num_samples: usize) -> SpatialResult<f64> {
    use crate::convex_hull::geometry::compute_bounding_box;
    use rand::Rng;

    if hull.vertex_indices.is_empty() {
        return Ok(0.0);
    }

    let ndim = hull.ndim();
    let (min_coords, max_coords) = compute_bounding_box(&hull.points.view(), &hull.vertex_indices);

    // Compute bounding box volume
    let mut bbox_volume = 1.0;
    for d in 0..ndim {
        let size = max_coords[d] - min_coords[d];
        if size <= 0.0 {
            return Ok(0.0); // Degenerate case
        }
        bbox_volume *= size;
    }

    // Monte Carlo sampling
    let mut inside_count = 0;
    let mut rng = rand::rng();

    for _ in 0..num_samples {
        let mut sample_point = vec![0.0; ndim];

        // Generate random point within bounding box
        for d in 0..ndim {
            sample_point[d] = rng.gen_range(min_coords[d]..max_coords[d]);
        }

        // Check if point is inside the convex hull
        if hull.contains(sample_point.as_slice()).unwrap_or(false) {
            inside_count += 1;
        }
    }

    // Estimate volume
    let fraction_inside = inside_count as f64 / num_samples as f64;
    let estimated_volume = fraction_inside * bbox_volume;

    Ok(estimated_volume)
}

/// Compute volume bounds (lower and upper estimates)
///
/// This function provides both lower and upper bounds for the volume,
/// which can be useful for validation or uncertainty quantification.
///
/// # Arguments
///
/// * `hull` - The convex hull
///
/// # Returns
///
/// * Result containing (lower_bound, upper_bound, exact_volume) tuple
///   where exact_volume is Some(value) if an exact computation was possible
///
/// # Examples
///
/// ```rust
/// use scirs2_spatial::convex_hull::ConvexHull;
/// use scirs2_spatial::convex_hull::properties::volume::compute_volume_bounds;
/// use ndarray::array;
///
/// let points = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
/// let hull = ConvexHull::new(&points.view()).unwrap();
/// let (lower, upper, exact) = compute_volume_bounds(&hull).unwrap();
/// assert!(exact.is_some());
/// assert!((exact.unwrap() - 1.0).abs() < 1e-10);
/// ```
pub fn compute_volume_bounds(hull: &ConvexHull) -> SpatialResult<(f64, f64, Option<f64>)> {
    // For low dimensions, we can compute exactly
    if hull.ndim() <= 3 {
        let exact = compute_volume(hull)?;
        return Ok((exact, exact, Some(exact)));
    }

    // For high dimensions, provide bounds
    let exact = compute_volume(hull).ok();

    if let Some(vol) = exact {
        Ok((vol, vol, Some(vol)))
    } else {
        // Use Monte Carlo with different sample sizes for bounds
        let lower_samples = 1000;
        let upper_samples = 10000;

        let lower_bound = compute_volume_monte_carlo(hull, lower_samples)?;
        let upper_bound = compute_volume_monte_carlo(hull, upper_samples)?;

        // Ensure bounds are properly ordered
        let (min_bound, max_bound) = if lower_bound <= upper_bound {
            (lower_bound * 0.9, upper_bound * 1.1) // Add some margin for uncertainty
        } else {
            (upper_bound * 0.9, lower_bound * 1.1)
        };

        Ok((min_bound, max_bound, None))
    }
}

/// Check if the volume computation is likely to be accurate
///
/// # Arguments
///
/// * `hull` - The convex hull
///
/// # Returns
///
/// * true if volume computation should be accurate, false otherwise
///
/// # Examples
///
/// ```rust
/// use scirs2_spatial::convex_hull::ConvexHull;
/// use scirs2_spatial::convex_hull::properties::volume::is_volume_computation_reliable;
/// use ndarray::array;
///
/// let points = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
/// let hull = ConvexHull::new(&points.view()).unwrap();
/// assert!(is_volume_computation_reliable(&hull));
/// ```
pub fn is_volume_computation_reliable(hull: &ConvexHull) -> bool {
    let ndim = hull.ndim();
    let nvertices = hull.vertex_indices.len();

    // 1D, 2D, 3D are generally reliable
    if ndim <= 3 {
        return true;
    }

    // For high dimensions, check if we have enough structure
    if hull.equations.is_some() && nvertices > ndim {
        return true;
    }

    // Very high dimensions with few vertices are unreliable
    if ndim > 10 && nvertices < 2 * ndim {
        return false;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convex_hull::ConvexHull;
    use ndarray::arr2;

    #[test]
    fn test_compute_2d_volume() {
        // Unit square
        let points = arr2(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        let hull = ConvexHull::new(&points.view()).unwrap();
        let area = compute_volume(&hull).unwrap();
        assert!((area - 1.0).abs() < 1e-10);

        // Triangle
        let points = arr2(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        let hull = ConvexHull::new(&points.view()).unwrap();
        let area = compute_volume(&hull).unwrap();
        assert!((area - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_compute_1d_volume() {
        // Line segment
        let points = arr2(&[[0.0], [3.0], [1.0], [2.0]]);
        let hull = ConvexHull::new(&points.view()).unwrap();
        let length = compute_volume(&hull).unwrap();
        assert!((length - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_volume_bounds() {
        let points = arr2(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        let hull = ConvexHull::new(&points.view()).unwrap();
        let (lower, upper, exact) = compute_volume_bounds(&hull).unwrap();

        assert!(exact.is_some());
        assert!((exact.unwrap() - 1.0).abs() < 1e-10);
        assert_eq!(lower, upper); // Should be exact for 2D
    }

    #[test]
    fn test_is_volume_computation_reliable() {
        // 2D case - should be reliable
        let points = arr2(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        let hull = ConvexHull::new(&points.view()).unwrap();
        assert!(is_volume_computation_reliable(&hull));

        // 3D case - should be reliable
        let points = arr2(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]);
        let hull = ConvexHull::new(&points.view()).unwrap();
        assert!(is_volume_computation_reliable(&hull));
    }

    #[test]
    fn test_compute_volume_monte_carlo() {
        let points = arr2(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        let hull = ConvexHull::new(&points.view()).unwrap();

        // Monte Carlo should give approximately correct result
        let estimated_area = compute_volume_monte_carlo(&hull, 10000).unwrap();
        assert!((estimated_area - 1.0).abs() < 0.1); // Allow for Monte Carlo error
    }

    #[test]
    #[ignore]
    fn test_degenerate_cases() {
        // Single point
        let points = arr2(&[[1.0, 2.0]]);
        let hull = ConvexHull::new(&points.view()).unwrap();
        let volume = compute_volume(&hull).unwrap();
        assert_eq!(volume, 0.0);

        // Two points (1D case)
        let points = arr2(&[[0.0], [5.0]]);
        let hull = ConvexHull::new(&points.view()).unwrap();
        let length = compute_volume(&hull).unwrap();
        assert!((length - 5.0).abs() < 1e-10);
    }
}
