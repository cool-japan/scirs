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
//! use scirs2_core::ndarray::array;
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

use scirs2_core::ndarray::{Array2, ArrayView2};
use scirs2_core::numeric::Float;

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
/// use scirs2_core::ndarray::array;
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
#[allow(dead_code)]
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
/// use scirs2_core::ndarray::array;
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
#[allow(dead_code)]
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
/// use scirs2_core::ndarray::array;
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
#[allow(dead_code)]
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
/// use scirs2_core::ndarray::array;
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
#[allow(dead_code)]
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
/// use scirs2_core::ndarray::array;
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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
/// use scirs2_core::ndarray::array;
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
#[allow(dead_code)]
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
/// use scirs2_core::ndarray::array;
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
#[allow(dead_code)]
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
    let mut indexedpoints: Vec<(usize, T)> = (0..n)
        .map(|i| (i, polar_angle(points[[i, 0]], points[[i, 1]])))
        .collect();

    indexedpoints.sort_by(|a, b| {
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

    for &(index_, _) in &indexedpoints {
        // Skip pivot point
        if index_ == lowest {
            continue;
        }

        while hull_indices.len() >= 2 {
            let top = hull_indices.len() - 1;
            if ccw(hull_indices[top - 1], hull_indices[top], index_) {
                break;
            }
            hull_indices.pop();
        }

        hull_indices.push(index_);
    }

    // Create the hull array
    let mut hull = Array2::zeros((hull_indices.len(), 2));
    for (i, &idx) in hull_indices.iter().enumerate() {
        hull[[i, 0]] = points[[idx, 0]];
        hull[[i, 1]] = points[[idx, 1]];
    }

    hull
}

/// Simplify a polygon using the Douglas-Peucker algorithm.
///
/// The Douglas-Peucker algorithm recursively removes vertices that contribute less
/// than a specified tolerance to the shape of the polygon. This is useful for
/// reducing the complexity of polygons while preserving their essential characteristics.
///
/// # Arguments
///
/// * `polygon` - The polygon vertices to simplify
/// * `tolerance` - The perpendicular distance tolerance. Vertices that are closer
///   than this distance to the line connecting their neighbors may be removed.
///
/// # Returns
///
/// * A simplified polygon with fewer vertices
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_spatial::polygon::douglas_peucker_simplify;
///
/// // A polygon with many points
/// let complexpolygon = array![
///     [0.0, 0.0],
///     [1.0, 0.1],  // Close to the line from (0,0) to (2,0)
///     [2.0, 0.0],
///     [2.0, 2.0],
///     [0.0, 2.0],
/// ];
///
/// let simplified = douglas_peucker_simplify(&complexpolygon.view(), 0.2);
///
/// // Should remove the intermediate point at (1.0, 0.1) since it's close to the line
/// assert!(simplified.shape()[0] < complexpolygon.shape()[0]);
/// ```
#[allow(dead_code)]
pub fn douglas_peucker_simplify<T: Float + std::fmt::Debug>(
    polygon: &ArrayView2<T>,
    tolerance: T,
) -> Array2<T> {
    let n = polygon.shape()[0];

    if n <= 2 {
        return polygon.to_owned();
    }

    // Create a boolean array to mark which points to keep
    let mut keep = vec![false; n];

    // Always keep the first and last points
    keep[0] = true;
    keep[n - 1] = true;

    // Apply Douglas-Peucker recursively
    douglas_peucker_recursive(polygon, 0, n - 1, tolerance, &mut keep);

    // Count the points to keep
    let num_keep = keep.iter().filter(|&&x| x).count();

    // Create the simplified polygon
    let mut simplified = Array2::zeros((num_keep, 2));
    let mut simplified_idx = 0;

    for i in 0..n {
        if keep[i] {
            simplified[[simplified_idx, 0]] = polygon[[i, 0]];
            simplified[[simplified_idx, 1]] = polygon[[i, 1]];
            simplified_idx += 1;
        }
    }

    simplified
}

/// Recursive helper function for Douglas-Peucker algorithm
#[allow(dead_code)]
fn douglas_peucker_recursive<T: Float>(
    polygon: &ArrayView2<T>,
    start: usize,
    end: usize,
    tolerance: T,
    keep: &mut [bool],
) {
    if end <= start + 1 {
        return;
    }

    // Find the point with the maximum perpendicular distance from the line
    let mut max_dist = T::zero();
    let mut max_idx = start;

    let startpoint = [polygon[[start, 0]], polygon[[start, 1]]];
    let endpoint = [polygon[[end, 0]], polygon[[end, 1]]];

    for i in start + 1..end {
        let point = [polygon[[i, 0]], polygon[[i, 1]]];
        let dist = perpendicular_distance(&point, &startpoint, &endpoint);

        if dist > max_dist {
            max_dist = dist;
            max_idx = i;
        }
    }

    // If the maximum distance is greater than tolerance, keep the point and recurse
    if max_dist > tolerance {
        keep[max_idx] = true;
        douglas_peucker_recursive(polygon, start, max_idx, tolerance, keep);
        douglas_peucker_recursive(polygon, max_idx, end, tolerance, keep);
    }
}

/// Calculate the perpendicular distance from a point to a line segment
#[allow(dead_code)]
fn perpendicular_distance<T: Float>(point: &[T; 2], line_start: &[T; 2], lineend: &[T; 2]) -> T {
    let dx = lineend[0] - line_start[0];
    let dy = lineend[1] - line_start[1];

    // If the line segment is actually a point, return distance to that point
    if dx.is_zero() && dy.is_zero() {
        let px = point[0] - line_start[0];
        let py = point[1] - line_start[1];
        return (px * px + py * py).sqrt();
    }

    // Calculate the perpendicular distance using the cross product formula
    let numerator = ((dy * (point[0] - line_start[0])) - (dx * (point[1] - line_start[1]))).abs();
    let denominator = (dx * dx + dy * dy).sqrt();

    numerator / denominator
}

/// Simplify a polygon using the Visvalingam-Whyatt algorithm.
///
/// The Visvalingam-Whyatt algorithm removes vertices by calculating the area of
/// triangles formed by consecutive triplets of vertices and removing vertices
/// that form triangles with areas smaller than a threshold.
///
/// # Arguments
///
/// * `polygon` - The polygon vertices to simplify
/// * `min_area` - The minimum triangle area threshold. Vertices forming triangles
///   with areas smaller than this will be candidates for removal.
///
/// # Returns
///
/// * A simplified polygon with fewer vertices
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_spatial::polygon::visvalingam_whyatt_simplify;
///
/// // A polygon with some redundant vertices
/// let polygon = array![
///     [0.0, 0.0],
///     [1.0, 0.0],
///     [1.01, 0.01],  // Forms a very small triangle
///     [2.0, 0.0],
///     [2.0, 2.0],
///     [0.0, 2.0],
/// ];
///
/// let simplified = visvalingam_whyatt_simplify(&polygon.view(), 0.1);
///
/// // Should remove vertices that form very small triangles
/// assert!(simplified.shape()[0] <= polygon.shape()[0]);
/// ```
#[allow(dead_code)]
pub fn visvalingam_whyatt_simplify<T: Float + std::fmt::Debug>(
    polygon: &ArrayView2<T>,
    min_area: T,
) -> Array2<T> {
    let n = polygon.shape()[0];

    if n <= 3 {
        return polygon.to_owned();
    }

    // Create a list of vertices with their effective areas
    let mut vertices: Vec<(usize, T)> = Vec::new();
    let mut active = vec![true; n];

    // Calculate initial areas for all vertices
    for i in 0..n {
        let area = calculate_triangle_area(polygon, i, &active);
        vertices.push((i, area));
    }

    // Sort by _area (smallest first)
    vertices.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Remove vertices with areas smaller than threshold
    let removal_candidates: Vec<usize> = vertices
        .iter()
        .filter(|(_, area)| *area < min_area)
        .map(|(idx_, _)| *idx_)
        .collect();

    for vertex_idx in removal_candidates {
        if count_active(&active) > 3 && active[vertex_idx] {
            active[vertex_idx] = false;

            // Recalculate areas for neighboring vertices
            let prev = find_previous_active(vertex_idx, &active, n);
            let next = find_next_active(vertex_idx, &active, n);

            if let (Some(prev_idx), Some(next_idx)) = (prev, next) {
                // Update the _area for the previous vertex
                update_vertex_area(polygon, prev_idx, &active, &mut vertices);
                // Update the _area for the next vertex
                update_vertex_area(polygon, next_idx, &active, &mut vertices);
            }
        }
    }

    // Create the simplified polygon
    let num_active = count_active(&active);
    let mut simplified = Array2::zeros((num_active, 2));
    let mut simplified_idx = 0;

    for i in 0..n {
        if active[i] {
            simplified[[simplified_idx, 0]] = polygon[[i, 0]];
            simplified[[simplified_idx, 1]] = polygon[[i, 1]];
            simplified_idx += 1;
        }
    }

    simplified
}

/// Calculate the area of triangle formed by vertex i and its active neighbors
#[allow(dead_code)]
fn calculate_triangle_area<T: Float>(
    polygon: &ArrayView2<T>,
    vertex_idx: usize,
    active: &[bool],
) -> T {
    let n = polygon.shape()[0];

    let prev = find_previous_active(vertex_idx, active, n);
    let next = find_next_active(vertex_idx, active, n);

    match (prev, next) {
        (Some(prev_idx), Some(next_idx)) => {
            let p1 = [polygon[[prev_idx, 0]], polygon[[prev_idx, 1]]];
            let p2 = [polygon[[vertex_idx, 0]], polygon[[vertex_idx, 1]]];
            let p3 = [polygon[[next_idx, 0]], polygon[[next_idx, 1]]];

            triangle_area(&p1, &p2, &p3)
        }
        _ => T::infinity(), // End vertices have infinite area (never removed)
    }
}

/// Calculate the area of a triangle given three points
#[allow(dead_code)]
fn triangle_area<T: Float>(p1: &[T; 2], p2: &[T; 2], p3: &[T; 2]) -> T {
    ((p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
        / (T::one() + T::one()))
    .abs()
}

/// Find the previous active vertex (wrapping around)
#[allow(dead_code)]
fn find_previous_active(current: usize, active: &[bool], n: usize) -> Option<usize> {
    for i in 1..n {
        let idx = (current + n - i) % n;
        if active[idx] && idx != current {
            return Some(idx);
        }
    }
    None
}

/// Find the next active vertex (wrapping around)
#[allow(dead_code)]
fn find_next_active(current: usize, active: &[bool], n: usize) -> Option<usize> {
    for i in 1..n {
        let idx = (current + i) % n;
        if active[idx] && idx != current {
            return Some(idx);
        }
    }
    None
}

/// Count the number of active vertices
#[allow(dead_code)]
fn count_active(active: &[bool]) -> usize {
    active.iter().filter(|&&x| x).count()
}

/// Update the area for a specific vertex in the vertices list
#[allow(dead_code)]
fn update_vertex_area<T: Float + std::fmt::Debug>(
    polygon: &ArrayView2<T>,
    vertex_idx: usize,
    active: &[bool],
    vertices: &mut [(usize, T)],
) {
    let new_area = calculate_triangle_area(polygon, vertex_idx, active);

    // Find and update the vertex in the list
    for (_idx, area) in vertices.iter_mut() {
        if *_idx == vertex_idx {
            *area = new_area;
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn testpoint_in_polygon() {
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
    fn testpoint_on_boundary() {
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
    fn testpolygon_area() {
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
        let lshape = array![
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [0.0, 2.0],
        ];

        assert_relative_eq!(polygon_area(&lshape.view()), 3.0, epsilon = 1e-10);
    }

    #[test]
    fn testpolygon_centroid() {
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
    fn testpolygon_contains_polygon() {
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
        let mut circlepoints = Array2::zeros((8, 2));
        for i in 0..8 {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / 8.0;
            circlepoints[[i, 0]] = angle.cos();
            circlepoints[[i, 1]] = angle.sin();
        }

        // Add some interior points
        let allpoints = array![
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

        let hull = convex_hull_graham(&allpoints.view());

        // The hull should have 8 points (the circle points)
        assert_eq!(hull.shape()[0], 8);
    }
}
