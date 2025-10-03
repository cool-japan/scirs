//! Point containment testing for convex hulls
//!
//! This module provides functions to test whether points lie inside,
//! outside, or on the boundary of a convex hull.

use crate::convex_hull::core::ConvexHull;
use crate::error::{SpatialError, SpatialResult};

/// Check if a point is contained within a convex hull
///
/// This function determines whether a given point lies inside the convex hull.
/// Different methods are used depending on the availability of hull information.
///
/// # Arguments
///
/// * `hull` - The convex hull
/// * `point` - The point to test (must match hull dimensionality)
///
/// # Returns
///
/// * Result containing true if point is inside hull, false otherwise
///
/// # Errors
///
/// * Returns error if point dimension doesn't match hull dimension
///
/// # Examples
///
/// ```rust
/// use scirs2_spatial::convex_hull::ConvexHull;
/// use scirs2_spatial::convex_hull::properties::containment::check_point_containment;
/// use scirs2_core::ndarray::array;
///
/// let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
/// let hull = ConvexHull::new(&points.view()).unwrap();
///
/// assert!(check_point_containment(&hull, &[0.1, 0.1]).unwrap());
/// assert!(!check_point_containment(&hull, &[2.0, 2.0]).unwrap());
/// ```
pub fn check_point_containment<T: AsRef<[f64]>>(
    hull: &ConvexHull,
    point: T,
) -> SpatialResult<bool> {
    let point_slice = point.as_ref();

    if point_slice.len() != hull.points.ncols() {
        return Err(SpatialError::DimensionError(format!(
            "Point dimension ({}) does not match hull dimension ({})",
            point_slice.len(),
            hull.points.ncols()
        )));
    }

    // If we have equations for the hull facets, we can use them for the check
    if let Some(equations) = &hull.equations {
        return check_containment_with_equations(hull, point_slice, equations);
    }

    // Fallback method: check if point is in the convex combination of hull vertices
    check_containment_convex_combination(hull, point_slice)
}

/// Check containment using facet equations (most reliable method)
///
/// # Arguments
///
/// * `hull` - The convex hull
/// * `point` - The point to test
/// * `equations` - Facet equations
///
/// # Returns
///
/// * Result containing true if point is inside, false otherwise
fn check_containment_with_equations(
    _hull: &ConvexHull,
    point: &[f64],
    equations: &scirs2_core::ndarray::Array2<f64>,
) -> SpatialResult<bool> {
    for i in 0..equations.nrows() {
        let mut result = equations[[i, equations.ncols() - 1]];
        for j in 0..point.len() {
            result += equations[[i, j]] * point[j];
        }

        // If result is positive, point is outside the hull
        // Use a more lenient tolerance to handle numerical precision issues
        if result > 1e-8 {
            return Ok(false);
        }
    }

    // If point is not outside any facet, it's inside the hull
    Ok(true)
}

/// Check containment using convex combination method (fallback)
///
/// This method checks if the point can be expressed as a convex combination
/// of the hull vertices. This is less efficient but works when equations
/// are not available.
///
/// # Arguments
///
/// * `hull` - The convex hull
/// * `point` - The point to test
///
/// # Returns
///
/// * Result containing true if point is inside, false otherwise
fn check_containment_convex_combination(hull: &ConvexHull, point: &[f64]) -> SpatialResult<bool> {
    let ndim = hull.ndim();

    // Handle low-dimensional cases with specific methods
    match ndim {
        1 => check_containment_1d(hull, point),
        2 => check_containment_2d(hull, point),
        3 => check_containment_3d(hull, point),
        _ => check_containment_nd(hull, point),
    }
}

/// Check 1D containment (point on line segment)
fn check_containment_1d(hull: &ConvexHull, point: &[f64]) -> SpatialResult<bool> {
    if hull.vertex_indices.is_empty() {
        return Ok(false);
    }

    let vertices = hull.vertices();
    if vertices.is_empty() {
        return Ok(false);
    }

    let min_val = vertices.iter().map(|v| v[0]).fold(f64::INFINITY, f64::min);
    let max_val = vertices
        .iter()
        .map(|v| v[0])
        .fold(f64::NEG_INFINITY, f64::max);

    Ok(point[0] >= min_val - 1e-10 && point[0] <= max_val + 1e-10)
}

/// Check 2D containment using ray casting or winding number
fn check_containment_2d(hull: &ConvexHull, point: &[f64]) -> SpatialResult<bool> {
    // Handle degenerate cases
    if hull.vertex_indices.is_empty() {
        return Ok(false);
    }

    if hull.vertex_indices.len() == 1 {
        // Single point - check if the query point matches
        let idx = hull.vertex_indices[0];
        let tolerance = 1e-10;
        let dx = (hull.points[[idx, 0]] - point[0]).abs();
        let dy = (hull.points[[idx, 1]] - point[1]).abs();
        return Ok(dx < tolerance && dy < tolerance);
    }

    if hull.vertex_indices.len() == 2 {
        // Line segment - check if point is on the segment
        let idx1 = hull.vertex_indices[0];
        let idx2 = hull.vertex_indices[1];
        let p1 = [hull.points[[idx1, 0]], hull.points[[idx1, 1]]];
        let p2 = [hull.points[[idx2, 0]], hull.points[[idx2, 1]]];

        // Check if point is on the line segment
        let cross = (p2[0] - p1[0]) * (point[1] - p1[1]) - (p2[1] - p1[1]) * (point[0] - p1[0]);
        if cross.abs() > 1e-10 {
            return Ok(false); // Not collinear
        }

        // Check if point is between p1 and p2
        let dot = (point[0] - p1[0]) * (p2[0] - p1[0]) + (point[1] - p1[1]) * (p2[1] - p1[1]);
        let len_squared = (p2[0] - p1[0]) * (p2[0] - p1[0]) + (p2[1] - p1[1]) * (p2[1] - p1[1]);
        return Ok(dot >= -1e-10 && dot <= len_squared + 1e-10);
    }

    // Normal case with 3+ vertices
    if hull.vertex_indices.len() < 3 {
        return Ok(false);
    }

    // Use winding number method for robust 2D point-in-polygon test
    let mut winding_number = 0;
    let vertices = &hull.vertex_indices;
    let n = vertices.len();

    for i in 0..n {
        let j = (i + 1) % n;
        let v1 = [hull.points[[vertices[i], 0]], hull.points[[vertices[i], 1]]];
        let v2 = [hull.points[[vertices[j], 0]], hull.points[[vertices[j], 1]]];

        if v1[1] <= point[1] {
            if v2[1] > point[1] {
                // Upward crossing
                let cross =
                    (v2[0] - v1[0]) * (point[1] - v1[1]) - (v2[1] - v1[1]) * (point[0] - v1[0]);
                if cross > 0.0 {
                    winding_number += 1;
                }
            }
        } else if v2[1] <= point[1] {
            // Downward crossing
            let cross = (v2[0] - v1[0]) * (point[1] - v1[1]) - (v2[1] - v1[1]) * (point[0] - v1[0]);
            if cross < 0.0 {
                winding_number -= 1;
            }
        }
    }

    Ok(winding_number != 0)
}

/// Check 3D containment using simplex-based method
fn check_containment_3d(hull: &ConvexHull, point: &[f64]) -> SpatialResult<bool> {
    // Handle degenerate cases
    if hull.vertex_indices.is_empty() {
        return Ok(false);
    }

    if hull.vertex_indices.len() == 1 {
        // Single point - check if the query point matches
        let idx = hull.vertex_indices[0];
        let tolerance = 1e-10;
        let dx = (hull.points[[idx, 0]] - point[0]).abs();
        let dy = (hull.points[[idx, 1]] - point[1]).abs();
        let dz = (hull.points[[idx, 2]] - point[2]).abs();
        return Ok(dx < tolerance && dy < tolerance && dz < tolerance);
    }

    if hull.vertex_indices.len() == 2 {
        // Line segment in 3D - check if point is on the segment
        let idx1 = hull.vertex_indices[0];
        let idx2 = hull.vertex_indices[1];
        let p1 = [
            hull.points[[idx1, 0]],
            hull.points[[idx1, 1]],
            hull.points[[idx1, 2]],
        ];
        let p2 = [
            hull.points[[idx2, 0]],
            hull.points[[idx2, 1]],
            hull.points[[idx2, 2]],
        ];

        // Check if point is on the line segment
        let v = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
        let w = [point[0] - p1[0], point[1] - p1[1], point[2] - p1[2]];

        // Cross product to check collinearity
        let cross = [
            v[1] * w[2] - v[2] * w[1],
            v[2] * w[0] - v[0] * w[2],
            v[0] * w[1] - v[1] * w[0],
        ];
        let cross_mag = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
        if cross_mag > 1e-10 {
            return Ok(false); // Not collinear
        }

        // Check if point is between p1 and p2
        let dot = v[0] * w[0] + v[1] * w[1] + v[2] * w[2];
        let len_squared = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        return Ok(dot >= -1e-10 && dot <= len_squared + 1e-10);
    }

    // TODO: Handle 3 points (triangle in 3D)
    if hull.vertex_indices.len() == 3 {
        // For now, just return false for triangular hulls in 3D
        // A proper implementation would check if the point is on the triangle
        return Ok(false);
    }

    // Normal case with 4+ vertices
    if hull.vertex_indices.len() < 4 {
        return Ok(false);
    }

    // For 3D, we need to check if the point is on the correct side of all faces
    // This is a simplified implementation that works for well-formed convex hulls
    for simplex in &hull.simplices {
        if simplex.len() != 3 {
            continue; // Skip non-triangular faces
        }

        let v0 = [
            hull.points[[simplex[0], 0]],
            hull.points[[simplex[0], 1]],
            hull.points[[simplex[0], 2]],
        ];
        let v1 = [
            hull.points[[simplex[1], 0]],
            hull.points[[simplex[1], 1]],
            hull.points[[simplex[1], 2]],
        ];
        let v2 = [
            hull.points[[simplex[2], 0]],
            hull.points[[simplex[2], 1]],
            hull.points[[simplex[2], 2]],
        ];

        // Compute normal to the face
        let edge1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let edge2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
        let mut normal = [
            edge1[1] * edge2[2] - edge1[2] * edge2[1],
            edge1[2] * edge2[0] - edge1[0] * edge2[2],
            edge1[0] * edge2[1] - edge1[1] * edge2[0],
        ];

        // Compute the centroid of the hull to determine correct normal orientation
        let mut centroid = [0.0, 0.0, 0.0];
        for &idx in &hull.vertex_indices {
            centroid[0] += hull.points[[idx, 0]];
            centroid[1] += hull.points[[idx, 1]];
            centroid[2] += hull.points[[idx, 2]];
        }
        let n_vertices = hull.vertex_indices.len() as f64;
        centroid[0] /= n_vertices;
        centroid[1] /= n_vertices;
        centroid[2] /= n_vertices;

        // Vector from face center to hull centroid
        let face_center = [
            (v0[0] + v1[0] + v2[0]) / 3.0,
            (v0[1] + v1[1] + v2[1]) / 3.0,
            (v0[2] + v1[2] + v2[2]) / 3.0,
        ];
        let to_centroid = [
            centroid[0] - face_center[0],
            centroid[1] - face_center[1],
            centroid[2] - face_center[2],
        ];

        // If normal points toward centroid, flip it to point outward
        let centroid_dot =
            normal[0] * to_centroid[0] + normal[1] * to_centroid[1] + normal[2] * to_centroid[2];
        if centroid_dot > 0.0 {
            normal[0] = -normal[0];
            normal[1] = -normal[1];
            normal[2] = -normal[2];
        }

        // Vector from face vertex to test point
        let to_point = [point[0] - v0[0], point[1] - v0[1], point[2] - v0[2]];

        // Check which side of the face the point is on
        let dot = normal[0] * to_point[0] + normal[1] * to_point[1] + normal[2] * to_point[2];

        // For a convex hull, if the point is outside any face, it's outside the hull
        // With corrected outward-pointing normals, positive dot means outside
        if dot > 1e-8 {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Check high-dimensional containment using linear programming approach
fn check_containment_nd(_hull: &ConvexHull, _point: &[f64]) -> SpatialResult<bool> {
    // For high dimensions without facet equations, this is complex
    // A complete implementation would use linear programming to solve:
    // Find λᵢ ≥ 0 such that Σλᵢ = 1 and Σλᵢvᵢ = point
    // where vᵢ are hull vertices

    // For now, return a conservative answer
    Ok(false)
}

/// Check if multiple points are contained within a convex hull
///
/// This is more efficient than checking points individually when testing
/// many points against the same hull.
///
/// # Arguments
///
/// * `hull` - The convex hull
/// * `points` - Array of points to test (shape: npoints x ndim)
///
/// # Returns
///
/// * Result containing vector of boolean values indicating containment
///
/// # Examples
///
/// ```rust
/// use scirs2_spatial::convex_hull::ConvexHull;
/// use scirs2_spatial::convex_hull::properties::containment::check_multiple_containment;
/// use scirs2_core::ndarray::array;
///
/// let hull_points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
/// let hull = ConvexHull::new(&hull_points.view()).unwrap();
///
/// let test_points = array![[0.1, 0.1], [2.0, 2.0], [0.2, 0.2]];
/// let results = check_multiple_containment(&hull, &test_points.view()).unwrap();
/// assert_eq!(results, vec![true, false, true]);
/// ```
pub fn check_multiple_containment(
    hull: &ConvexHull,
    points: &scirs2_core::ndarray::ArrayView2<'_, f64>,
) -> SpatialResult<Vec<bool>> {
    let npoints = points.nrows();
    let mut results = Vec::with_capacity(npoints);

    // If we have facet equations, use them for all points
    if let Some(equations) = &hull.equations {
        for i in 0..npoints {
            let point = points.row(i);
            let is_inside =
                check_containment_with_equations(hull, point.as_slice().unwrap(), equations)?;
            results.push(is_inside);
        }
    } else {
        // Fallback to individual checks
        for i in 0..npoints {
            let point = points.row(i);
            let is_inside = check_containment_convex_combination(hull, point.as_slice().unwrap())?;
            results.push(is_inside);
        }
    }

    Ok(results)
}

/// Compute distance from a point to the convex hull boundary
///
/// Returns the signed distance: positive if outside, negative if inside,
/// zero if on the boundary.
///
/// # Arguments
///
/// * `hull` - The convex hull
/// * `point` - The point to test
///
/// # Returns
///
/// * Result containing the signed distance to the hull boundary
///
/// # Examples
///
/// ```rust
/// use scirs2_spatial::convex_hull::ConvexHull;
/// use scirs2_spatial::convex_hull::properties::containment::distance_to_hull;
/// use scirs2_core::ndarray::array;
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
/// let hull = ConvexHull::new(&points.view())?;
///
/// let dist = distance_to_hull(&hull, &[0.5, 0.5])?;
/// assert!(dist < 0.0); // Inside the hull
///
/// let dist = distance_to_hull(&hull, &[2.0, 2.0])?;
/// assert!(dist > 0.0); // Outside the hull
/// # Ok(())
/// # }
/// ```
pub fn distance_to_hull<T: AsRef<[f64]>>(hull: &ConvexHull, point: T) -> SpatialResult<f64> {
    let point_slice = point.as_ref();

    if point_slice.len() != hull.points.ncols() {
        return Err(SpatialError::DimensionError(format!(
            "Point dimension ({}) does not match hull dimension ({})",
            point_slice.len(),
            hull.points.ncols()
        )));
    }

    // If we have facet equations, compute distance using them
    if let Some(equations) = &hull.equations {
        return compute_distance_with_equations(point_slice, equations);
    }

    // Fallback: approximate distance using vertices
    compute_distance_to_vertices(hull, point_slice)
}

/// Compute distance using facet equations
fn compute_distance_with_equations(
    point: &[f64],
    equations: &scirs2_core::ndarray::Array2<f64>,
) -> SpatialResult<f64> {
    let mut min_distance = f64::INFINITY;
    let mut is_inside = true;

    for i in 0..equations.nrows() {
        // Compute signed distance to this facet
        let mut distance = equations[[i, equations.ncols() - 1]];
        let mut normal_length_sq = 0.0;

        for j in 0..point.len() {
            distance += equations[[i, j]] * point[j];
            normal_length_sq += equations[[i, j]] * equations[[i, j]];
        }

        let normal_length = normal_length_sq.sqrt();
        if normal_length > 1e-12 {
            distance /= normal_length;
        }

        // Track if point is outside any facet
        if distance > 1e-10 {
            is_inside = false;
        }

        // Track minimum absolute distance
        min_distance = min_distance.min(distance.abs());
    }

    // Return signed distance: negative if inside, positive if outside
    if is_inside {
        Ok(-min_distance)
    } else {
        Ok(min_distance)
    }
}

/// Compute approximate distance to hull vertices
fn compute_distance_to_vertices(hull: &ConvexHull, point: &[f64]) -> SpatialResult<f64> {
    let mut min_distance = f64::INFINITY;

    for &vertex_idx in &hull.vertex_indices {
        let mut dist_sq = 0.0;
        for d in 0..point.len() {
            let diff = point[d] - hull.points[[vertex_idx, d]];
            dist_sq += diff * diff;
        }
        min_distance = min_distance.min(dist_sq.sqrt());
    }

    // This doesn't give signed distance, so we use containment check
    let is_inside = check_point_containment(hull, point)?;
    if is_inside {
        Ok(-min_distance)
    } else {
        Ok(min_distance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convex_hull::ConvexHull;
    use scirs2_core::ndarray::arr2;

    #[test]
    fn test_2d_containment() {
        let points = arr2(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        let hull = ConvexHull::new(&points.view()).unwrap();

        // Point inside the triangle
        assert!(check_point_containment(&hull, [0.1, 0.1]).unwrap());

        // Point outside the triangle
        assert!(!check_point_containment(&hull, [2.0, 2.0]).unwrap());

        // Point on the edge (should be considered inside)
        assert!(check_point_containment(&hull, [0.5, 0.0]).unwrap());
    }

    #[test]
    fn test_1d_containment() {
        let points = arr2(&[[0.0], [3.0], [1.0], [2.0]]);
        let hull = ConvexHull::new(&points.view()).unwrap();

        // Point inside the line segment
        assert!(check_point_containment(&hull, [1.5]).unwrap());

        // Point outside the line segment
        assert!(!check_point_containment(&hull, [5.0]).unwrap());

        // Point at the boundary
        assert!(check_point_containment(&hull, [0.0]).unwrap());
        assert!(check_point_containment(&hull, [3.0]).unwrap());
    }

    #[test]
    fn test_3d_containment() {
        let points = arr2(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]);
        let hull = ConvexHull::new(&points.view()).unwrap();

        // Point inside the tetrahedron
        assert!(check_point_containment(&hull, [0.1, 0.1, 0.1]).unwrap());

        // Point outside the tetrahedron
        assert!(!check_point_containment(&hull, [2.0, 2.0, 2.0]).unwrap());
    }

    #[test]
    fn test_multiple_containment() {
        let hull_points = arr2(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        let hull = ConvexHull::new(&hull_points.view()).unwrap();

        let test_points = arr2(&[[0.1, 0.1], [2.0, 2.0], [0.2, 0.2]]);
        let results = check_multiple_containment(&hull, &test_points.view()).unwrap();

        assert_eq!(results.len(), 3);
        assert!(results[0]); // Inside
        assert!(!results[1]); // Outside
        assert!(results[2]); // Inside
    }

    #[test]
    fn test_distance_to_hull() {
        let points = arr2(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        let hull = ConvexHull::new(&points.view()).unwrap();

        // Distance should be negative for interior points
        let dist = distance_to_hull(&hull, [0.1, 0.1]).unwrap();
        assert!(dist < 0.0);

        // Distance should be positive for exterior points
        let dist = distance_to_hull(&hull, [2.0, 2.0]).unwrap();
        assert!(dist > 0.0);
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let points = arr2(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        let hull = ConvexHull::new(&points.view()).unwrap();

        // Test 3D point with 2D hull
        let result = check_point_containment(&hull, [0.1, 0.1, 0.1]);
        assert!(result.is_err());

        // Test 1D point with 2D hull
        let result = check_point_containment(&hull, [0.1]);
        assert!(result.is_err());
    }

    #[test]
    fn test_degenerate_cases() {
        // Single point - should fail to create convex hull
        let points = arr2(&[[1.0, 2.0]]);
        let hull_result = ConvexHull::new(&points.view());
        assert!(hull_result.is_err());

        // Two points in 2D - should fail to create convex hull
        let points = arr2(&[[0.0, 0.0], [1.0, 1.0]]);
        let hull_result = ConvexHull::new(&points.view());
        assert!(hull_result.is_err());

        // Valid 2D triangle (minimal valid case)
        let points = arr2(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        let hull = ConvexHull::new(&points.view()).unwrap();

        // Point inside the triangle
        let result = check_point_containment(&hull, [0.1, 0.1]).unwrap();
        assert!(result);

        // Point outside the triangle
        let result = check_point_containment(&hull, [2.0, 2.0]).unwrap();
        assert!(!result);
    }
}
