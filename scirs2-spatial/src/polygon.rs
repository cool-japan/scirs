//! Polygon operations module
//!
//! This module provides functionality for working with polygons, including:
//! - Point in polygon testing
//! - Area and centroid calculations
//! - Polygon operations (simplification, transformation)
//!
//! # Examples
//!
//! ```
//! use ndarray::array;
//! use scirs2_spatial::polygon::point_in_polygon;
//!
//! // Create a square polygon
//! let polygon = array![
//!     [0.0, 0.0],
//!     [1.0, 0.0],
//!     [1.0, 1.0],
//!     [0.0, 1.0],
//! ];
//!
//! // Test if a point is inside
//! let inside = point_in_polygon(&[0.5, 0.5], &polygon.view());
//! assert!(inside);
//!
//! // Test if a point is outside
//! let outside = point_in_polygon(&[1.5, 0.5], &polygon.view());
//! assert!(!outside);
//! ```

use ndarray::{Array2, ArrayView2};
use num_traits::Float;

/// Tests if a point is inside a polygon using the ray casting algorithm.
///
/// The algorithm works by casting a ray from the test point to infinity in the positive x direction,
/// and counting how many times the ray intersects the polygon boundary. If the count is odd,
/// the point is inside the polygon; if even, the point is outside.
///
/// # Arguments
///
/// * `point` - The point to test, as [x, y] coordinates
/// * `polygon` - The polygon vertices as an array of [x, y] coordinates, assumed to be in order
///
/// # Returns
///
/// * `true` if the point is inside the polygon, `false` otherwise
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_spatial::polygon::point_in_polygon;
///
/// // Create a square polygon
/// let polygon = array![
///     [0.0, 0.0],
///     [1.0, 0.0],
///     [1.0, 1.0],
///     [0.0, 1.0],
/// ];
///
/// // Test if a point is inside
/// let inside = point_in_polygon(&[0.5, 0.5], &polygon.view());
/// assert!(inside);
///
/// // Test if a point is outside
/// let outside = point_in_polygon(&[1.5, 0.5], &polygon.view());
/// assert!(!outside);
/// ```
pub fn point_in_polygon<T: Float>(point: &[T], polygon: &ArrayView2<T>) -> bool {
    let x = point[0];
    let y = point[1];
    let n = polygon.shape()[0];

    if n < 3 {
        return false; // A polygon must have at least 3 vertices
    }

    // First check if the point is on the boundary
    let epsilon = T::from(1e-10).unwrap();
    if point_on_boundary(point, polygon, epsilon) {
        return true;
    }

    // Ray casting algorithm - count intersections of a horizontal ray with polygon edges
    let mut inside = false;

    for i in 0..n {
        let j = (i + 1) % n;

        let vi_y = polygon[[i, 1]];
        let vj_y = polygon[[j, 1]];

        // Check if the edge crosses the horizontal line at y
        if ((vi_y <= y) && (vj_y > y)) || ((vi_y > y) && (vj_y <= y)) {
            // Compute x-coordinate of intersection
            let vi_x = polygon[[i, 0]];
            let vj_x = polygon[[j, 0]];

            // Check if intersection is to the right of the point
            let slope = (vj_x - vi_x) / (vj_y - vi_y);
            let intersect_x = vi_x + (y - vi_y) * slope;

            if x < intersect_x {
                inside = !inside;
            }
        }
    }

    inside
}

/// Tests if a point is on the boundary of a polygon.
///
/// A point is considered on the boundary if it's within epsilon distance
/// to any edge of the polygon.
///
/// # Arguments
///
/// * `point` - The point to test, as [x, y] coordinates
/// * `polygon` - The polygon vertices as an array of [x, y] coordinates
/// * `epsilon` - Distance threshold for considering a point on the boundary
///
/// # Returns
///
/// * `true` if the point is on the polygon boundary, `false` otherwise
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_spatial::polygon::point_on_boundary;
///
/// // Create a square polygon
/// let polygon = array![
///     [0.0, 0.0],
///     [1.0, 0.0],
///     [1.0, 1.0],
///     [0.0, 1.0],
/// ];
///
/// // Test if a point is on the boundary
/// let on_boundary = point_on_boundary(&[0.0, 0.5], &polygon.view(), 1e-10);
/// assert!(on_boundary);
///
/// // Test if a point is not on the boundary
/// let not_on_boundary = point_on_boundary(&[0.5, 0.5], &polygon.view(), 1e-10);
/// assert!(!not_on_boundary);
/// ```
pub fn point_on_boundary<T: Float>(point: &[T], polygon: &ArrayView2<T>, epsilon: T) -> bool {
    let x = point[0];
    let y = point[1];
    let n = polygon.shape()[0];

    if n < 2 {
        return false;
    }

    for i in 0..n {
        let j = (i + 1) % n;

        let x1 = polygon[[i, 0]];
        let y1 = polygon[[i, 1]];
        let x2 = polygon[[j, 0]];
        let y2 = polygon[[j, 1]];

        // Calculate distance from point to line segment
        let length_squared = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);

        if length_squared.is_zero() {
            // The segment is a point
            let dist = ((x - x1) * (x - x1) + (y - y1) * (y - y1)).sqrt();
            if dist < epsilon {
                return true;
            }
        } else {
            // Calculate the projection of the point onto the line
            let t = ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / length_squared;

            if t < T::zero() {
                // Closest point is the start of the segment
                let dist = ((x - x1) * (x - x1) + (y - y1) * (y - y1)).sqrt();
                if dist < epsilon {
                    return true;
                }
            } else if t > T::one() {
                // Closest point is the end of the segment
                let dist = ((x - x2) * (x - x2) + (y - y2) * (y - y2)).sqrt();
                if dist < epsilon {
                    return true;
                }
            } else {
                // Closest point is along the segment
                let projection_x = x1 + t * (x2 - x1);
                let projection_y = y1 + t * (y2 - y1);
                let dist = ((x - projection_x) * (x - projection_x)
                    + (y - projection_y) * (y - projection_y))
                    .sqrt();
                if dist < epsilon {
                    return true;
                }
            }
        }
    }

    false
}

/// Calculate the area of a simple polygon.
///
/// Uses the Shoelace formula (also known as the surveyor's formula).
///
/// # Arguments
///
/// * `polygon` - The polygon vertices as an array of [x, y] coordinates
///
/// # Returns
///
/// * The area of the polygon
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_spatial::polygon::polygon_area;
///
/// // Create a 1x1 square
/// let square = array![
///     [0.0, 0.0],
///     [1.0, 0.0],
///     [1.0, 1.0],
///     [0.0, 1.0],
/// ];
///
/// let area: f64 = polygon_area(&square.view());
/// assert!((area - 1.0).abs() < 1e-10);
/// ```
pub fn polygon_area<T: Float>(polygon: &ArrayView2<T>) -> T {
    let n = polygon.shape()[0];

    if n < 3 {
        return T::zero(); // A polygon must have at least 3 vertices
    }

    let mut area = T::zero();

    for i in 0..n {
        let j = (i + 1) % n;

        let xi = polygon[[i, 0]];
        let yi = polygon[[i, 1]];
        let xj = polygon[[j, 0]];
        let yj = polygon[[j, 1]];

        area = area + (xi * yj - xj * yi);
    }

    (area / (T::one() + T::one())).abs()
}

/// Calculate the centroid of a simple polygon.
///
/// # Arguments
///
/// * `polygon` - The polygon vertices as an array of [x, y] coordinates
///
/// # Returns
///
/// * The centroid coordinates as [x, y]
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_spatial::polygon::polygon_centroid;
///
/// // Create a 1x1 square
/// let square = array![
///     [0.0, 0.0],
///     [1.0, 0.0],
///     [1.0, 1.0],
///     [0.0, 1.0],
/// ];
///
/// let centroid: Vec<f64> = polygon_centroid(&square.view());
/// assert!((centroid[0] - 0.5).abs() < 1e-10);
/// assert!((centroid[1] - 0.5).abs() < 1e-10);
/// ```
pub fn polygon_centroid<T: Float>(polygon: &ArrayView2<T>) -> Vec<T> {
    let n = polygon.shape()[0];

    if n < 3 {
        // Return the average of points for degenerate cases
        let mut x_sum = T::zero();
        let mut y_sum = T::zero();

        for i in 0..n {
            x_sum = x_sum + polygon[[i, 0]];
            y_sum = y_sum + polygon[[i, 1]];
        }

        if n > 0 {
            return vec![x_sum / T::from(n).unwrap(), y_sum / T::from(n).unwrap()];
        } else {
            return vec![T::zero(), T::zero()];
        }
    }

    let mut cx = T::zero();
    let mut cy = T::zero();
    let mut area = T::zero();

    for i in 0..n {
        let j = (i + 1) % n;

        let xi = polygon[[i, 0]];
        let yi = polygon[[i, 1]];
        let xj = polygon[[j, 0]];
        let yj = polygon[[j, 1]];

        let cross = xi * yj - xj * yi;

        cx = cx + (xi + xj) * cross;
        cy = cy + (yi + yj) * cross;
        area = area + cross;
    }

    area = area / (T::one() + T::one());

    if area.is_zero() {
        // Degenerate case, return the average of points
        let mut x_sum = T::zero();
        let mut y_sum = T::zero();

        for i in 0..n {
            x_sum = x_sum + polygon[[i, 0]];
            y_sum = y_sum + polygon[[i, 1]];
        }

        return vec![x_sum / T::from(n).unwrap(), y_sum / T::from(n).unwrap()];
    }

    let six = T::from(6).unwrap();
    cx = cx / (six * area);
    cy = cy / (six * area);

    // The formula can give negative coordinates if the polygon is
    // oriented clockwise, so we take the absolute value
    vec![cx.abs(), cy.abs()]
}

/// Tests if polygon A contains polygon B (every point of B is inside or on the boundary of A).
///
/// # Arguments
///
/// * `polygon_a` - The container polygon vertices
/// * `polygon_b` - The contained polygon vertices
///
/// # Returns
///
/// * `true` if polygon A contains polygon B, `false` otherwise
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_spatial::polygon::polygon_contains_polygon;
///
/// // Create an outer polygon (large square)
/// let outer = array![
///     [0.0, 0.0],
///     [2.0, 0.0],
///     [2.0, 2.0],
///     [0.0, 2.0],
/// ];
///
/// // Create an inner polygon (small square)
/// let inner = array![
///     [0.5, 0.5],
///     [1.5, 0.5],
///     [1.5, 1.5],
///     [0.5, 1.5],
/// ];
///
/// let contains = polygon_contains_polygon(&outer.view(), &inner.view());
/// assert!(contains);
/// ```
pub fn polygon_contains_polygon<T: Float>(
    polygon_a: &ArrayView2<T>,
    polygon_b: &ArrayView2<T>,
) -> bool {
    // Special case: if A and B are the same polygon, A contains B
    if polygon_a.shape() == polygon_b.shape() {
        let n = polygon_a.shape()[0];
        let mut same = true;

        for i in 0..n {
            if polygon_a[[i, 0]] != polygon_b[[i, 0]] || polygon_a[[i, 1]] != polygon_b[[i, 1]] {
                same = false;
                break;
            }
        }

        if same {
            return true;
        }
    }

    let n_b = polygon_b.shape()[0];

    // Check if all vertices of polygon B are inside or on the boundary of polygon A
    for i in 0..n_b {
        let point = [polygon_b[[i, 0]], polygon_b[[i, 1]]];
        if !point_in_polygon(&point, polygon_a) {
            return false;
        }
    }

    // Check for non-coincident edge intersections only if polygons aren't identical
    if polygon_a.shape() != polygon_b.shape() {
        let n_a = polygon_a.shape()[0];

        // Check if any edges of B cross any edges of A (that would mean B is not contained)
        for i in 0..n_a {
            let j = (i + 1) % n_a;
            let a1 = [polygon_a[[i, 0]], polygon_a[[i, 1]]];
            let a2 = [polygon_a[[j, 0]], polygon_a[[j, 1]]];

            for k in 0..n_b {
                let l = (k + 1) % n_b;
                let b1 = [polygon_b[[k, 0]], polygon_b[[k, 1]]];
                let b2 = [polygon_b[[l, 0]], polygon_b[[l, 1]]];

                // Only consider proper intersections (not edge overlaps)
                if segments_intersect(&a1, &a2, &b1, &b2)
                    && !segments_overlap(&a1, &a2, &b1, &b2, T::epsilon())
                    && !point_on_boundary(&a1, polygon_b, T::epsilon())
                    && !point_on_boundary(&a2, polygon_b, T::epsilon())
                    && !point_on_boundary(&b1, polygon_a, T::epsilon())
                    && !point_on_boundary(&b2, polygon_a, T::epsilon())
                {
                    return false;
                }
            }
        }
    }

    true
}

/// Check if two line segments intersect.
///
/// # Arguments
///
/// * `a1`, `a2` - The endpoints of the first segment
/// * `b1`, `b2` - The endpoints of the second segment
///
/// # Returns
///
/// * `true` if the segments intersect, `false` otherwise
fn segments_intersect<T: Float>(a1: &[T], a2: &[T], b1: &[T], b2: &[T]) -> bool {
    // Function to compute orientation of triplet (p, q, r)
    // Returns:
    // 0 -> collinear
    // 1 -> clockwise
    // 2 -> counterclockwise
    let orientation = |p: &[T], q: &[T], r: &[T]| -> i32 {
        let val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1]);

        if val < T::zero() {
            return 1; // clockwise
        } else if val > T::zero() {
            return 2; // counterclockwise
        }

        0 // collinear
    };

    // Function to check if point q is on segment pr
    let on_segment = |p: &[T], q: &[T], r: &[T]| -> bool {
        q[0] <= p[0].max(r[0])
            && q[0] >= p[0].min(r[0])
            && q[1] <= p[1].max(r[1])
            && q[1] >= p[1].min(r[1])
    };

    let o1 = orientation(a1, a2, b1);
    let o2 = orientation(a1, a2, b2);
    let o3 = orientation(b1, b2, a1);
    let o4 = orientation(b1, b2, a2);

    // General case
    if o1 != o2 && o3 != o4 {
        return true;
    }

    // Special cases
    if o1 == 0 && on_segment(a1, b1, a2) {
        return true;
    }

    if o2 == 0 && on_segment(a1, b2, a2) {
        return true;
    }

    if o3 == 0 && on_segment(b1, a1, b2) {
        return true;
    }

    if o4 == 0 && on_segment(b1, a2, b2) {
        return true;
    }

    false
}

/// Check if two line segments overlap (share multiple points).
///
/// # Arguments
///
/// * `a1`, `a2` - The endpoints of the first segment
/// * `b1`, `b2` - The endpoints of the second segment
/// * `epsilon` - Tolerance for floating-point comparisons
///
/// # Returns
///
/// * `true` if the segments overlap, `false` otherwise
fn segments_overlap<T: Float>(a1: &[T], a2: &[T], b1: &[T], b2: &[T], epsilon: T) -> bool {
    // Check if the segments are collinear
    let cross = (a2[0] - a1[0]) * (b2[1] - b1[1]) - (a2[1] - a1[1]) * (b2[0] - b1[0]);

    if cross.abs() > epsilon {
        return false; // Not collinear
    }

    // Check if the segments overlap on the x-axis
    let overlap_x = !(a2[0] < b1[0].min(b2[0]) - epsilon || a1[0] > b1[0].max(b2[0]) + epsilon);

    // Check if the segments overlap on the y-axis
    let overlap_y = !(a2[1] < b1[1].min(b2[1]) - epsilon || a1[1] > b1[1].max(b2[1]) + epsilon);

    overlap_x && overlap_y
}

/// Check if a polygon is simple (non-self-intersecting).
///
/// # Arguments
///
/// * `polygon` - The polygon vertices
///
/// # Returns
///
/// * `true` if the polygon is simple, `false` otherwise
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_spatial::polygon::is_simple_polygon;
///
/// // A simple square
/// let simple = array![
///     [0.0, 0.0],
///     [1.0, 0.0],
///     [1.0, 1.0],
///     [0.0, 1.0],
/// ];
///
/// // A self-intersecting "bow tie"
/// let complex = array![
///     [0.0, 0.0],
///     [1.0, 1.0],
///     [0.0, 1.0],
///     [1.0, 0.0],
/// ];
///
/// assert!(is_simple_polygon(&simple.view()));
/// assert!(!is_simple_polygon(&complex.view()));
/// ```
pub fn is_simple_polygon<T: Float>(polygon: &ArrayView2<T>) -> bool {
    let n = polygon.shape()[0];

    if n < 3 {
        return true; // Degenerate cases are considered simple
    }

    // Check each pair of non-adjacent edges for intersection
    for i in 0..n {
        let i1 = (i + 1) % n;

        let a1 = [polygon[[i, 0]], polygon[[i, 1]]];
        let a2 = [polygon[[i1, 0]], polygon[[i1, 1]]];

        for j in i + 2..i + n - 1 {
            let j_mod = j % n;
            let j1 = (j_mod + 1) % n;

            // Skip edges that share a vertex
            if j1 == i || j_mod == i1 {
                continue;
            }

            let b1 = [polygon[[j_mod, 0]], polygon[[j_mod, 1]]];
            let b2 = [polygon[[j1, 0]], polygon[[j1, 1]]];

            if segments_intersect(&a1, &a2, &b1, &b2) {
                return false; // Self-intersection found
            }
        }
    }

    true
}

/// Compute the convex hull of a polygon using the Graham scan algorithm.
///
/// # Arguments
///
/// * `points` - The input points
///
/// # Returns
///
/// * A new array containing the convex hull vertices
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_spatial::polygon::convex_hull_graham;
///
/// // A set of points
/// let points = array![
///     [0.0, 0.0],
///     [1.0, 0.0],
///     [0.5, 0.5],  // Inside point
///     [1.0, 1.0],
///     [0.0, 1.0],
/// ];
///
/// let hull = convex_hull_graham(&points.view());
///
/// // The hull should have only 4 points (the corners)
/// assert_eq!(hull.shape()[0], 4);
/// ```
pub fn convex_hull_graham<T: Float + std::fmt::Debug>(points: &ArrayView2<T>) -> Array2<T> {
    let n = points.shape()[0];

    if n <= 3 {
        // For 3 or fewer points, all points are on the convex hull
        return points.to_owned();
    }

    // Find the point with the lowest y-coordinate (and leftmost if tied)
    let mut lowest = 0;
    for i in 1..n {
        if points[[i, 1]] < points[[lowest, 1]]
            || (points[[i, 1]] == points[[lowest, 1]] && points[[i, 0]] < points[[lowest, 0]])
        {
            lowest = i;
        }
    }

    // Pivot point
    let pivot_x = points[[lowest, 0]];
    let pivot_y = points[[lowest, 1]];

    // Function to compute polar angle of a point relative to the pivot
    let polar_angle = |x: T, y: T| -> T {
        let dx = x - pivot_x;
        let dy = y - pivot_y;

        // Handle special case where points are the same
        if dx.is_zero() && dy.is_zero() {
            return T::neg_infinity();
        }

        // Use atan2 for the polar angle
        dy.atan2(dx)
    };

    // Sort points by polar angle
    let mut indexed_points: Vec<(usize, T)> = (0..n)
        .map(|i| (i, polar_angle(points[[i, 0]], points[[i, 1]])))
        .collect();

    indexed_points.sort_by(|a, b| {
        // First by polar angle
        let angle_cmp = a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal);

        if angle_cmp == std::cmp::Ordering::Equal {
            // Break ties by distance from pivot
            let dist_a =
                (points[[a.0, 0]] - pivot_x).powi(2) + (points[[a.0, 1]] - pivot_y).powi(2);
            let dist_b =
                (points[[b.0, 0]] - pivot_x).powi(2) + (points[[b.0, 1]] - pivot_y).powi(2);
            dist_a
                .partial_cmp(&dist_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        } else {
            angle_cmp
        }
    });

    // Function to check if three points make a right turn
    let ccw = |i1: usize, i2: usize, i3: usize| -> bool {
        let p1_x = points[[i1, 0]];
        let p1_y = points[[i1, 1]];
        let p2_x = points[[i2, 0]];
        let p2_y = points[[i2, 1]];
        let p3_x = points[[i3, 0]];
        let p3_y = points[[i3, 1]];

        let val = (p2_x - p1_x) * (p3_y - p1_y) - (p2_y - p1_y) * (p3_x - p1_x);
        val > T::zero()
    };

    // Graham scan algorithm
    let mut hull_indices = Vec::new();

    // Add first three points
    hull_indices.push(lowest);

    for &(index, _) in &indexed_points {
        // Skip pivot point
        if index == lowest {
            continue;
        }

        while hull_indices.len() >= 2 {
            let top = hull_indices.len() - 1;
            if ccw(hull_indices[top - 1], hull_indices[top], index) {
                break;
            }
            hull_indices.pop();
        }

        hull_indices.push(index);
    }

    // Create the hull array
    let mut hull = Array2::zeros((hull_indices.len(), 2));
    for (i, &idx) in hull_indices.iter().enumerate() {
        hull[[i, 0]] = points[[idx, 0]];
        hull[[i, 1]] = points[[idx, 1]];
    }

    hull
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_point_in_polygon() {
        // Simple square
        let square = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],];

        // Points inside
        assert!(point_in_polygon(&[0.5, 0.5], &square.view()));
        assert!(point_in_polygon(&[0.1, 0.1], &square.view()));
        assert!(point_in_polygon(&[0.9, 0.9], &square.view()));

        // Points outside
        assert!(!point_in_polygon(&[1.5, 0.5], &square.view()));
        assert!(!point_in_polygon(&[-0.5, 0.5], &square.view()));
        assert!(!point_in_polygon(&[0.5, 1.5], &square.view()));
        assert!(!point_in_polygon(&[0.5, -0.5], &square.view()));

        // Points on boundary (considered inside)
        assert!(point_in_polygon(&[0.0, 0.5], &square.view()));
        assert!(point_in_polygon(&[1.0, 0.5], &square.view()));
        assert!(point_in_polygon(&[0.5, 0.0], &square.view()));
        assert!(point_in_polygon(&[0.5, 1.0], &square.view()));

        // More complex polygon (concave)
        let concave = array![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [1.0, 1.0], [0.0, 2.0],];

        // Inside the concave part - this point is actually outside the polygon
        // Modified test to assert the correct result
        assert!(!point_in_polygon(&[1.0, 1.5], &concave.view()));

        // Inside the convex part
        assert!(point_in_polygon(&[0.5, 0.5], &concave.view()));

        // Outside (but inside the bounding box)
        // For the given concave shape, point (1.5, 0.5) is actually inside
        assert!(point_in_polygon(&[1.5, 0.5], &concave.view()));
    }

    #[test]
    fn test_point_on_boundary() {
        // Simple square
        let square = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],];

        let epsilon = 1e-10;

        // Points on boundary
        assert!(point_on_boundary(&[0.0, 0.5], &square.view(), epsilon));
        assert!(point_on_boundary(&[1.0, 0.5], &square.view(), epsilon));
        assert!(point_on_boundary(&[0.5, 0.0], &square.view(), epsilon));
        assert!(point_on_boundary(&[0.5, 1.0], &square.view(), epsilon));

        // Corner points
        assert!(point_on_boundary(&[0.0, 0.0], &square.view(), epsilon));
        assert!(point_on_boundary(&[1.0, 0.0], &square.view(), epsilon));
        assert!(point_on_boundary(&[1.0, 1.0], &square.view(), epsilon));
        assert!(point_on_boundary(&[0.0, 1.0], &square.view(), epsilon));

        // Points not on boundary
        assert!(!point_on_boundary(&[0.5, 0.5], &square.view(), epsilon));
        assert!(!point_on_boundary(&[1.5, 0.5], &square.view(), epsilon));
        assert!(!point_on_boundary(&[0.0, 2.0], &square.view(), epsilon));

        // Points near but not exactly on boundary should still be detected
        // with a larger epsilon
        assert!(point_on_boundary(&[0.0, 0.5 + 1e-5], &square.view(), 1e-4));
    }

    #[test]
    fn test_polygon_area() {
        // Square with area 1
        let square = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],];

        assert_relative_eq!(polygon_area(&square.view()), 1.0, epsilon = 1e-10);

        // Rectangle with area 6
        let rectangle = array![[0.0, 0.0], [3.0, 0.0], [3.0, 2.0], [0.0, 2.0],];

        assert_relative_eq!(polygon_area(&rectangle.view()), 6.0, epsilon = 1e-10);

        // Triangle with area 0.5
        let triangle = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0],];

        assert_relative_eq!(polygon_area(&triangle.view()), 0.5, epsilon = 1e-10);

        // L-shaped polygon
        let l_shape = array![
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [0.0, 2.0],
        ];

        assert_relative_eq!(polygon_area(&l_shape.view()), 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_polygon_centroid() {
        // Square with centroid at (0.5, 0.5)
        let square = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],];

        let centroid = polygon_centroid(&square.view());
        assert_relative_eq!(centroid[0], 0.5, epsilon = 1e-10);
        assert_relative_eq!(centroid[1], 0.5, epsilon = 1e-10);

        // Rectangle with centroid at (1.5, 1.0)
        let rectangle = array![[0.0, 0.0], [3.0, 0.0], [3.0, 2.0], [0.0, 2.0],];

        let centroid = polygon_centroid(&rectangle.view());
        assert_relative_eq!(centroid[0], 1.5, epsilon = 1e-10);
        assert_relative_eq!(centroid[1], 1.0, epsilon = 1e-10);

        // Triangle with centroid at (1/3, 1/3)
        let triangle = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0],];

        let centroid = polygon_centroid(&triangle.view());
        assert_relative_eq!(centroid[0], 1.0 / 3.0, epsilon = 1e-10);
        assert_relative_eq!(centroid[1], 1.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_polygon_contains_polygon() {
        // Outer square
        let outer = array![[0.0, 0.0], [3.0, 0.0], [3.0, 3.0], [0.0, 3.0],];

        // Inner square
        let inner = array![[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0],];

        // Outer should contain inner
        assert!(polygon_contains_polygon(&outer.view(), &inner.view()));

        // Inner should not contain outer
        assert!(!polygon_contains_polygon(&inner.view(), &outer.view()));

        // Overlapping squares
        let overlap = array![[2.0, 2.0], [4.0, 2.0], [4.0, 4.0], [2.0, 4.0],];

        // Neither should fully contain the other
        assert!(!polygon_contains_polygon(&outer.view(), &overlap.view()));
        assert!(!polygon_contains_polygon(&overlap.view(), &outer.view()));

        // A polygon contains itself
        assert!(polygon_contains_polygon(&outer.view(), &outer.view()));
    }

    #[test]
    fn test_is_simple_polygon() {
        // Simple square (non-self-intersecting)
        let square = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],];

        assert!(is_simple_polygon(&square.view()));

        // Self-intersecting bow tie
        let bowtie = array![[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0],];

        assert!(!is_simple_polygon(&bowtie.view()));

        // More complex non-self-intersecting polygon
        let complex = array![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [1.0, 1.0], [0.0, 2.0],];

        assert!(is_simple_polygon(&complex.view()));
    }

    #[test]
    fn test_convex_hull_graham() {
        // Simple test with a square and a point inside
        let points = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.5], // This point should not be in the hull
            [1.0, 1.0],
            [0.0, 1.0],
        ];

        let hull = convex_hull_graham(&points.view());

        // The hull should have only 4 points (the corners)
        assert_eq!(hull.shape()[0], 4);

        // Check that the inside point is not in the hull
        let mut found_inside = false;
        for i in 0..hull.shape()[0] {
            if (hull[[i, 0]] - 0.5).abs() < 1e-10 && (hull[[i, 1]] - 0.5).abs() < 1e-10 {
                found_inside = true;
                break;
            }
        }
        assert!(!found_inside);

        // Test with points in a circle-like arrangement
        let mut circle_points = Array2::zeros((8, 2));
        for i in 0..8 {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / 8.0;
            circle_points[[i, 0]] = angle.cos();
            circle_points[[i, 1]] = angle.sin();
        }

        // Add some interior points
        let all_points = array![
            [0.0, 0.0],  // Center point
            [0.5, 0.0],  // Interior point
            [0.0, 0.5],  // Interior point
            [-0.5, 0.0], // Interior point
            [0.0, -0.5], // Interior point
            // Add the circle points
            [1.0, 0.0],
            [
                std::f64::consts::FRAC_1_SQRT_2,
                std::f64::consts::FRAC_1_SQRT_2
            ],
            [0.0, 1.0],
            [
                -std::f64::consts::FRAC_1_SQRT_2,
                std::f64::consts::FRAC_1_SQRT_2
            ],
            [-1.0, 0.0],
            [
                -std::f64::consts::FRAC_1_SQRT_2,
                -std::f64::consts::FRAC_1_SQRT_2
            ],
            [0.0, -1.0],
            [
                std::f64::consts::FRAC_1_SQRT_2,
                -std::f64::consts::FRAC_1_SQRT_2
            ],
        ];

        let hull = convex_hull_graham(&all_points.view());

        // The hull should have 8 points (the circle points)
        assert_eq!(hull.shape()[0], 8);
    }
}
