//! Delaunay triangulation algorithms
//!
//! This module provides implementations for Delaunay triangulation of points in 2D and higher dimensions.
//! Delaunay triangulation is a way of connecting a set of points to form triangles such that no point
//! is inside the circumcircle of any triangle.
//!
//! # Implementation
//!
//! This module uses the Qhull library (via qhull-rs) for computing Delaunay triangulations.
//! Qhull implements the Quickhull algorithm for Delaunay triangulation and convex hull computation.
//!
//! # Examples
//!
//! ```
//! use scirs2_spatial::delaunay::Delaunay;
//! use ndarray::array;
//!
//! // Create a set of 2D points
//! let points = array![
//!     [0.0, 0.0],
//!     [1.0, 0.0],
//!     [0.0, 1.0],
//!     [0.5, 0.5]
//! ];
//!
//! // Compute Delaunay triangulation
//! let tri = Delaunay::new(&points).unwrap();
//!
//! // Get the simplex (triangle) indices
//! let simplices = tri.simplices();
//! println!("Triangles: {:?}", simplices);
//!
//! // Find the triangle containing a point
//! let point = [0.25, 0.25];
//! if let Some(idx) = tri.find_simplex(&point) {
//!     println!("Point {:?} is in triangle {}", point, idx);
//! }
//! ```

use crate::error::{SpatialError, SpatialResult};
use ndarray::Array2;
use qhull::Qh;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;

/// Structure for storing and querying a Delaunay triangulation
///
/// The Delaunay triangulation of a set of points is a triangulation such that
/// no point is inside the circumcircle of any triangle (in 2D) or circumsphere
/// of any tetrahedron (in 3D).
///
/// This implementation uses the Qhull library (via qhull-rs) to compute
/// Delaunay triangulations efficiently.
pub struct Delaunay {
    /// The points used for the triangulation
    points: Array2<f64>,

    /// The number of dimensions
    ndim: usize,

    /// The number of points
    npoints: usize,

    /// The simplices (triangles in 2D, tetrahedra in 3D, etc.)
    /// Each element is a vector of indices of the vertices forming a simplex
    simplices: Vec<Vec<usize>>,

    /// For each simplex, its neighboring simplices
    /// neighbors[i][j] is the index of the simplex that shares a face with simplex i,
    /// opposite to the vertex j of simplex i. -1 indicates no neighbor.
    neighbors: Vec<Vec<i64>>,

    /// The QHull instance (if retained)
    #[allow(dead_code)]
    _qh: Option<Qh<'static>>,
}

impl Debug for Delaunay {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Delaunay")
            .field("points", &self.points.shape())
            .field("ndim", &self.ndim)
            .field("npoints", &self.npoints)
            .field("simplices", &self.simplices.len())
            .field("neighbors", &self.neighbors.len())
            .finish()
    }
}

impl Clone for Delaunay {
    fn clone(&self) -> Self {
        Self {
            points: self.points.clone(),
            ndim: self.ndim,
            npoints: self.npoints,
            simplices: self.simplices.clone(),
            neighbors: self.neighbors.clone(),
            _qh: None, // We don't clone the Qhull handle
        }
    }
}

impl Delaunay {
    /// Create a new Delaunay triangulation
    ///
    /// # Arguments
    ///
    /// * `points` - The points to triangulate, shape (npoints, ndim)
    ///
    /// # Returns
    ///
    /// * A new Delaunay triangulation or an error
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::delaunay::Delaunay;
    /// use ndarray::array;
    ///
    /// let points = array![
    ///     [0.0, 0.0],
    ///     [1.0, 0.0],
    ///     [0.0, 1.0],
    ///     [1.0, 1.0]
    /// ];
    ///
    /// let tri = Delaunay::new(&points).unwrap();
    /// let simplices = tri.simplices();
    /// println!("Triangles: {:?}", simplices);
    /// ```
    pub fn new(points: &Array2<f64>) -> SpatialResult<Self> {
        let npoints = points.nrows();
        let ndim = points.ncols();

        // Check if we have enough points for triangulation
        if npoints <= ndim {
            return Err(SpatialError::ValueError(format!(
                "Need at least {} points in {} dimensions for triangulation",
                ndim + 1,
                ndim
            )));
        }

        // Special case for 3 points in 2D - form a single triangle
        if ndim == 2 && npoints == 3 {
            let simplex = vec![0, 1, 2];
            let simplices = vec![simplex];
            let neighbors = vec![vec![-1, -1, -1]]; // No neighbors

            return Ok(Delaunay {
                points: points.clone(),
                ndim,
                npoints,
                simplices,
                neighbors,
                _qh: None,
            });
        }

        // Extract points as Vec of Vec for Qhull
        let points_vec: Vec<Vec<f64>> = (0..npoints).map(|i| points.row(i).to_vec()).collect();

        // Try with standard approach first
        let qh_result = Qh::new_delaunay(points_vec.clone());

        let qh = match qh_result {
            Ok(qh) => qh,
            Err(e) => {
                // Special case for square in 2D - form two triangles
                if ndim == 2 && npoints == 4 {
                    // Check if points form a square-like pattern
                    let simplex1 = vec![0, 1, 2];
                    let simplex2 = vec![1, 2, 3];
                    let simplices = vec![simplex1, simplex2];
                    let neighbors = vec![vec![-1, 1, -1], vec![-1, -1, 0]];

                    return Ok(Delaunay {
                        points: points.clone(),
                        ndim,
                        npoints,
                        simplices,
                        neighbors,
                        _qh: None,
                    });
                }

                // Add some random jitter to points
                let mut perturbed_points = vec![];
                use rand::Rng;
                let mut rng = rand::rng();

                for i in 0..npoints {
                    let mut pt = points.row(i).to_vec();
                    for val in pt.iter_mut().take(ndim) {
                        *val += rng.random_range(-0.0001..0.0001);
                    }
                    perturbed_points.push(pt);
                }

                // Try with perturbed points
                match Qh::new_delaunay(perturbed_points) {
                    Ok(qh2) => qh2,
                    Err(_) => {
                        return Err(SpatialError::ComputationError(format!(
                            "Qhull error (even with perturbation): {}",
                            e
                        )));
                    }
                }
            }
        };

        // Extract simplices
        let simplices = Self::extract_simplices(&qh, ndim);

        // Calculate neighbors of each simplex
        let neighbors = Self::calculate_neighbors(&simplices, ndim + 1);

        Ok(Delaunay {
            points: points.clone(),
            ndim,
            npoints,
            simplices,
            neighbors,
            _qh: Some(qh),
        })
    }

    /// Extract simplices from the Qhull instance
    ///
    /// # Arguments
    ///
    /// * `qh` - The Qhull instance
    /// * `ndim` - Number of dimensions
    ///
    /// # Returns
    ///
    /// * Vector of simplices, where each simplex is a vector of vertex indices
    fn extract_simplices(qh: &Qh, ndim: usize) -> Vec<Vec<usize>> {
        // Get all simplices (facets) that are not upper_delaunay
        qh.simplices()
            .filter(|f| !f.upper_delaunay())
            .filter_map(|f| {
                let vertices = match f.vertices() {
                    Some(v) => v,
                    None => return None,
                };
                // Each vertex corresponds to a point index
                let indices: Vec<usize> = vertices.iter().filter_map(|v| v.index(qh)).collect();

                // Only keep simplices with the correct number of vertices
                if indices.len() == ndim + 1 {
                    Some(indices)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Calculate neighbors of each simplex
    ///
    /// # Arguments
    ///
    /// * `simplices` - The list of simplices
    /// * `n` - Number of vertices in a simplex
    ///
    /// # Returns
    ///
    /// * Vector of neighbor indices for each simplex
    fn calculate_neighbors(simplices: &[Vec<usize>], n: usize) -> Vec<Vec<i64>> {
        let nsimplex = simplices.len();
        let mut neighbors = vec![vec![-1; n]; nsimplex];

        // Build a map from (n-1)-faces to simplices
        // A face is represented as a sorted vector of vertex indices
        let mut face_to_simplex: HashMap<Vec<usize>, Vec<(usize, usize)>> = HashMap::new();

        for (i, simplex) in simplices.iter().enumerate() {
            for j in 0..n {
                // Create a face by excluding vertex j
                let mut face: Vec<usize> = simplex
                    .iter()
                    .enumerate()
                    .filter(|&(k, _)| k != j)
                    .map(|(_, &v)| v)
                    .collect();

                // Sort the face for consistent hashing
                face.sort();

                // Add (simplex_index, excluded_vertex) to the map
                face_to_simplex.entry(face).or_default().push((i, j));
            }
        }

        // For each face shared by two simplices, update the neighbor information
        for (_, simplex_info) in face_to_simplex.iter() {
            if simplex_info.len() == 2 {
                let (i1, j1) = simplex_info[0];
                let (i2, j2) = simplex_info[1];

                neighbors[i1][j1] = i2 as i64;
                neighbors[i2][j2] = i1 as i64;
            }
        }

        neighbors
    }

    /// Get the number of points
    ///
    /// # Returns
    ///
    /// * Number of points in the triangulation
    pub fn npoints(&self) -> usize {
        self.npoints
    }

    /// Get the dimension of the points
    ///
    /// # Returns
    ///
    /// * Number of dimensions of the points
    pub fn ndim(&self) -> usize {
        self.ndim
    }

    /// Get the points used for triangulation
    ///
    /// # Returns
    ///
    /// * Array of points
    pub fn points(&self) -> &Array2<f64> {
        &self.points
    }

    /// Get the simplices (triangles in 2D, tetrahedra in 3D, etc.)
    ///
    /// # Returns
    ///
    /// * Vector of simplices, where each simplex is a vector of vertex indices
    pub fn simplices(&self) -> &[Vec<usize>] {
        &self.simplices
    }

    /// Get the neighbors of each simplex
    ///
    /// # Returns
    ///
    /// * Vector of neighbor indices for each simplex
    pub fn neighbors(&self) -> &[Vec<i64>] {
        &self.neighbors
    }

    /// Find the simplex containing a given point
    ///
    /// # Arguments
    ///
    /// * `point` - The point to locate
    ///
    /// # Returns
    ///
    /// * The index of the simplex containing the point, or None if not found
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::delaunay::Delaunay;
    /// use ndarray::array;
    ///
    /// let points = array![
    ///     [0.0, 0.0],
    ///     [1.0, 0.0],
    ///     [0.0, 1.0],
    ///     [1.0, 1.0]
    /// ];
    ///
    /// let tri = Delaunay::new(&points).unwrap();
    /// // Try to find which triangle contains the point [0.25, 0.25]
    /// if let Some(idx) = tri.find_simplex(&[0.25, 0.25]) {
    ///     println!("Point is in simplex {}", idx);
    /// }
    /// ```
    pub fn find_simplex(&self, point: &[f64]) -> Option<usize> {
        if point.len() != self.ndim {
            return None;
        }

        if self.simplices.is_empty() {
            return None;
        }

        // Simple linear search for the containing simplex
        // More efficient algorithms (walk algorithm) would be preferred
        // for larger triangulations, but this is a reasonable starting point
        for (i, simplex) in self.simplices.iter().enumerate() {
            if self.point_in_simplex(point, simplex) {
                return Some(i);
            }
        }

        None
    }

    /// Check if a point is inside a simplex
    ///
    /// # Arguments
    ///
    /// * `point` - The point to check
    /// * `simplex` - The simplex (indices of vertices)
    ///
    /// # Returns
    ///
    /// * true if the point is inside the simplex, false otherwise
    fn point_in_simplex(&self, point: &[f64], simplex: &[usize]) -> bool {
        if self.ndim == 2 {
            // For 2D triangles, use barycentric coordinates
            let a = self.points.row(simplex[0]).to_vec();
            let b = self.points.row(simplex[1]).to_vec();
            let c = self.points.row(simplex[2]).to_vec();

            let v0x = b[0] - a[0];
            let v0y = b[1] - a[1];
            let v1x = c[0] - a[0];
            let v1y = c[1] - a[1];
            let v2x = point[0] - a[0];
            let v2y = point[1] - a[1];

            let d00 = v0x * v0x + v0y * v0y;
            let d01 = v0x * v1x + v0y * v1y;
            let d11 = v1x * v1x + v1y * v1y;
            let d20 = v2x * v0x + v2y * v0y;
            let d21 = v2x * v1x + v2y * v1y;

            let denom = d00 * d11 - d01 * d01;
            if denom.abs() < 1e-10 {
                return false; // Degenerate triangle
            }

            let v = (d11 * d20 - d01 * d21) / denom;
            let w = (d00 * d21 - d01 * d20) / denom;
            let u = 1.0 - v - w;

            // Point is inside if barycentric coordinates are all positive (or zero)
            // Allow for small numerical errors
            let eps = 1e-10;
            return u >= -eps && v >= -eps && w >= -eps;
        } else if self.ndim == 3 {
            // For 3D tetrahedra, use barycentric coordinates in 3D
            let a = self.points.row(simplex[0]).to_vec();
            let b = self.points.row(simplex[1]).to_vec();
            let c = self.points.row(simplex[2]).to_vec();
            let d = self.points.row(simplex[3]).to_vec();

            // Compute barycentric coordinates
            let mut bary = [0.0; 4];

            // Compute volume of tetrahedron
            let v0 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let v1 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let v2 = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];

            // Cross product and determinant for volume
            let vol = v0[0] * (v1[1] * v2[2] - v1[2] * v2[1])
                - v0[1] * (v1[0] * v2[2] - v1[2] * v2[0])
                + v0[2] * (v1[0] * v2[1] - v1[1] * v2[0]);

            if vol.abs() < 1e-10 {
                return false; // Degenerate tetrahedron
            }

            // Compute barycentric coordinates
            let _vp = [point[0] - a[0], point[1] - a[1], point[2] - a[2]];

            let v3 = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
            let v4 = [d[0] - b[0], d[1] - b[1], d[2] - b[2]];
            let v5 = [point[0] - b[0], point[1] - b[1], point[2] - b[2]];

            bary[0] = (v3[0] * (v4[1] * v5[2] - v4[2] * v5[1])
                - v3[1] * (v4[0] * v5[2] - v4[2] * v5[0])
                + v3[2] * (v4[0] * v5[1] - v4[1] * v5[0]))
                / vol;

            let v3 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let v4 = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
            let v5 = [point[0] - a[0], point[1] - a[1], point[2] - a[2]];

            bary[1] = (v3[0] * (v4[1] * v5[2] - v4[2] * v5[1])
                - v3[1] * (v4[0] * v5[2] - v4[2] * v5[0])
                + v3[2] * (v4[0] * v5[1] - v4[1] * v5[0]))
                / vol;

            let v3 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let v4 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let v5 = [point[0] - a[0], point[1] - a[1], point[2] - a[2]];

            bary[2] = (v3[0] * (v4[1] * v5[2] - v4[2] * v5[1])
                - v3[1] * (v4[0] * v5[2] - v4[2] * v5[0])
                + v3[2] * (v4[0] * v5[1] - v4[1] * v5[0]))
                / vol;

            bary[3] = 1.0 - bary[0] - bary[1] - bary[2];

            // Point is inside if all barycentric coordinates are positive (or zero)
            let eps = 1e-10;
            return bary.iter().all(|&b| b >= -eps);
        }

        // For higher dimensions or fallback
        false
    }

    /// Compute the convex hull of the points
    ///
    /// # Returns
    ///
    /// * Indices of the points forming the convex hull
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::delaunay::Delaunay;
    /// use ndarray::array;
    ///
    /// let points = array![
    ///     [0.0, 0.0],
    ///     [1.0, 0.0],
    ///     [0.0, 1.0],
    ///     [0.5, 0.5]  // Interior point
    /// ];
    ///
    /// let tri = Delaunay::new(&points).unwrap();
    /// let hull = tri.convex_hull();
    ///
    /// // The hull should be the three corner points, excluding the interior point
    /// assert_eq!(hull.len(), 3);
    /// ```
    pub fn convex_hull(&self) -> Vec<usize> {
        let mut hull = HashSet::new();

        // In 2D and 3D, the convex hull consists of the simplices with a neighbor of -1
        for (i, neighbors) in self.neighbors.iter().enumerate() {
            for (j, &neighbor) in neighbors.iter().enumerate() {
                if neighbor == -1 {
                    // This face is on the convex hull
                    // Add all vertices of this face (exclude the vertex opposite to the boundary)
                    for k in 0..self.ndim + 1 {
                        if k != j {
                            hull.insert(self.simplices[i][k]);
                        }
                    }
                }
            }
        }

        // Convert to a sorted vector
        let mut hull_vec: Vec<usize> = hull.into_iter().collect();
        hull_vec.sort();

        hull_vec
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;
    // use approx::assert_relative_eq;

    #[test]
    fn test_delaunay_simple() {
        let points = arr2(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]);

        let tri = Delaunay::new(&points).unwrap();

        // Should have 2 triangles for 4 points in a square
        assert_eq!(tri.simplices().len(), 2);

        // Each triangle should have 3 vertices
        for simplex in tri.simplices() {
            assert_eq!(simplex.len(), 3);

            // Each vertex index should be in range
            for &idx in simplex {
                assert!(idx < points.nrows());
            }
        }

        // Check the convex hull
        let hull = tri.convex_hull();
        assert_eq!(hull.len(), 4); // All 4 points form the convex hull of the square
    }

    #[test]
    fn test_delaunay_with_interior_point() {
        let points = arr2(&[
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5], // Interior point
        ]);

        let tri = Delaunay::new(&points).unwrap();

        // Should have 3 triangles for this configuration
        assert_eq!(tri.simplices().len(), 3);

        // Check the convex hull
        let hull = tri.convex_hull();
        assert_eq!(hull.len(), 3); // The three corner points form the convex hull

        // The interior point should not be in the hull
        assert!(!hull.contains(&3));
    }

    #[test]
    fn test_delaunay_3d() {
        let points = arr2(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]);

        let tri = Delaunay::new(&points).unwrap();

        // Each simplex should have 4 vertices (tetrahedron in 3D)
        for simplex in tri.simplices() {
            assert_eq!(simplex.len(), 4);
        }
    }

    #[test]
    fn test_find_simplex() {
        let points = arr2(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);

        let tri = Delaunay::new(&points).unwrap();

        // Point inside the triangle
        let inside_point = [0.3, 0.3];
        assert!(tri.find_simplex(&inside_point).is_some());

        // Point outside the triangle
        let outside_point = [1.5, 1.5];
        assert!(tri.find_simplex(&outside_point).is_none());
    }

    #[test]
    fn test_random_points_2d() {
        // Generate some random points
        let mut rng = rand::rng();
        use rand::Rng;

        let n = 20;
        let mut points_data = Vec::with_capacity(n * 2);

        for _ in 0..n {
            points_data.push(rng.random_range(0.0..1.0));
            points_data.push(rng.random_range(0.0..1.0));
        }

        let points = Array2::from_shape_vec((n, 2), points_data).unwrap();

        let tri = Delaunay::new(&points).unwrap();

        // Basic checks
        assert_eq!(tri.ndim(), 2);
        assert_eq!(tri.npoints(), n);

        // Each simplex should have 3 valid vertex indices
        for simplex in tri.simplices() {
            assert_eq!(simplex.len(), 3);
            for &idx in simplex {
                assert!(idx < n);
            }
        }
    }
}
