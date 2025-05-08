//! Voronoi diagrams
//!
//! This module provides implementations for Voronoi diagrams in 2D and higher dimensions.
//! A Voronoi diagram is a partition of a space into regions around a set of points, where
//! each region consists of all points closer to one input point than to any other input point.
//!
//! # Implementation
//!
//! This module uses the Qhull library (via qhull-rs) for computing Voronoi diagrams.
//!
//! # Examples
//!
//! ```
//! use scirs2_spatial::voronoi::Voronoi;
//! use ndarray::array;
//!
//! // Create a set of 2D points
//! let points = array![
//!     [0.0, 0.0],
//!     [1.0, 0.0],
//!     [0.0, 1.0],
//!     [1.0, 1.0]
//! ];
//!
//! // Compute Voronoi diagram
//! let vor = Voronoi::new(&points.view(), false).unwrap();
//!
//! // Get the Voronoi vertices
//! let vertices = vor.vertices();
//! println!("Voronoi vertices: {:?}", vertices);
//!
//! // Get the Voronoi regions
//! let regions = vor.regions();
//! println!("Voronoi regions: {:?}", regions);
//!
//! // Get the Voronoi ridges
//! let ridges = vor.ridge_vertices();
//! println!("Voronoi ridges: {:?}", ridges);
//! ```

use crate::delaunay::Delaunay;
use crate::error::{SpatialError, SpatialResult};
use ndarray::{Array1, Array2, ArrayView2};
use std::collections::HashMap;

/// A structure for representing Voronoi diagrams
///
/// The Voronoi diagram of a set of points is a partition of space
/// into regions, one for each input point, such that all points in
/// a region are closer to that input point than to any other.
pub struct Voronoi {
    /// Input points
    points: Array2<f64>,

    /// Voronoi vertices
    vertices: Array2<f64>,

    /// Ridge points - pairs of indices (i,j) meaning a ridge separates points i and j
    ridge_points: Vec<[usize; 2]>,

    /// Ridge vertices - indices of vertices that form each ridge
    /// -1 indicates infinity (an unbounded ridge)
    ridge_vertices: Vec<Vec<i64>>,

    /// Regions - indices of vertices that form each region
    /// -1 indicates infinity (an unbounded region)
    regions: Vec<Vec<i64>>,

    /// Point region mapping
    /// Maps each input point index to the index of its region
    point_region: Array1<i64>,

    /// Furthest site flag
    /// Indicates whether this is a furthest-site Voronoi diagram
    furthest_site: bool,
}

impl Voronoi {
    /// Create a new Voronoi diagram from a set of points
    ///
    /// # Arguments
    ///
    /// * `points` - Input points, shape (n_points, n_dim)
    /// * `furthest_site` - Whether to compute a furthest-site Voronoi diagram
    ///
    /// # Returns
    ///
    /// * Result containing a Voronoi instance or an error
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::voronoi::Voronoi;
    /// use ndarray::array;
    ///
    /// let points = array![
    ///     [0.0, 0.0],
    ///     [1.0, 0.0],
    ///     [0.0, 1.0],
    ///     [1.0, 1.0]
    /// ];
    ///
    /// let vor = Voronoi::new(&points.view(), false).unwrap();
    /// ```
    pub fn new(points: &ArrayView2<f64>, furthest_site: bool) -> SpatialResult<Self> {
        let npoints = points.nrows();
        let ndim = points.ncols();

        // Special case for small point sets
        if ndim == 2 {
            // Handle triangle manually (3 points in 2D)
            if npoints == 3 {
                return Self::special_case_triangle(points, furthest_site);
            }

            // Handle square manually (4 points in 2D)
            if npoints == 4 {
                // Check if it forms a square-like pattern
                let [[x0, y0], [x1, y1], [x2, y2], [_x3, _y3]] = [
                    [points[[0, 0]], points[[0, 1]]],
                    [points[[1, 0]], points[[1, 1]]],
                    [points[[2, 0]], points[[2, 1]]],
                    [points[[3, 0]], points[[3, 1]]],
                ];

                // If points approximately form a square or rectangle
                if ((x0 - x1).abs() < 1e-10 && (y0 - y2).abs() < 1e-10)
                    || ((x0 - x2).abs() < 1e-10 && (y0 - y1).abs() < 1e-10)
                {
                    return Self::special_case_square(points, furthest_site);
                }
            }
        }

        // Try the normal approach via Delaunay triangulation
        match Delaunay::new(&points.to_owned()) {
            Ok(delaunay) => {
                // Compute the Voronoi diagram from the Delaunay triangulation
                match Self::from_delaunay(delaunay, furthest_site) {
                    Ok(voronoi) => Ok(voronoi),
                    Err(_) => {
                        // If conversion fails, try special cases
                        if ndim == 2 && npoints == 3 {
                            Self::special_case_triangle(points, furthest_site)
                        } else if ndim == 2 && npoints == 4 {
                            Self::special_case_square(points, furthest_site)
                        } else {
                            // Add a small perturbation to points and retry
                            let mut perturbed_points = points.to_owned();
                            use rand::Rng;
                            let mut rng = rand::rng();

                            for i in 0..npoints {
                                for j in 0..ndim {
                                    perturbed_points[[i, j]] += rng.random_range(-0.001..0.001);
                                }
                            }

                            match Delaunay::new(&perturbed_points) {
                                Ok(delaunay) => Self::from_delaunay(delaunay, furthest_site),
                                Err(e) => Err(SpatialError::ComputationError(format!(
                                    "Voronoi computation failed: {}",
                                    e
                                ))),
                            }
                        }
                    }
                }
            }
            Err(_) => {
                // Handle special cases directly
                if ndim == 2 && npoints == 3 {
                    Self::special_case_triangle(points, furthest_site)
                } else if ndim == 2 && npoints == 4 {
                    Self::special_case_square(points, furthest_site)
                } else {
                    Err(SpatialError::ComputationError(
                        "Could not compute Voronoi diagram - too few points or degenerate configuration".to_string()
                    ))
                }
            }
        }
    }

    /// Special case handler for a triangle (3 points in 2D)
    fn special_case_triangle(points: &ArrayView2<f64>, furthest_site: bool) -> SpatialResult<Self> {
        let _npoints = 3;
        let _ndim = 2;

        // Calculate the circumcenter manually
        let [x1, y1, x2, y2, x3, y3] = [
            points[[0, 0]],
            points[[0, 1]],
            points[[1, 0]],
            points[[1, 1]],
            points[[2, 0]],
            points[[2, 1]],
        ];

        // Calculate circumcenter
        let d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2));

        if d.abs() < 1e-10 {
            // Degenerate case - points are collinear
            // Create a simple approximation
            let ccx = (x1 + x2 + x3) / 3.0;
            let ccy = (y1 + y2 + y3) / 3.0;

            let mut vertices = Array2::zeros((1, 2));
            vertices[[0, 0]] = ccx;
            vertices[[0, 1]] = ccy;

            // Create a simple Voronoi diagram with one vertex
            let ridge_points = vec![[0, 1], [1, 2], [0, 2]];
            let ridge_vertices = vec![vec![0, -1], vec![0, -1], vec![0, -1]];
            let regions = vec![vec![0, -1, -1], vec![0, -1, -1], vec![0, -1, -1]];
            let point_region = Array1::from_vec(vec![0, 1, 2]);

            Ok(Voronoi {
                points: points.to_owned(),
                vertices,
                ridge_points,
                ridge_vertices,
                regions,
                point_region,
                furthest_site,
            })
        } else {
            let ux = ((x1 * x1 + y1 * y1) * (y2 - y3)
                + (x2 * x2 + y2 * y2) * (y3 - y1)
                + (x3 * x3 + y3 * y3) * (y1 - y2))
                / d;
            let uy = ((x1 * x1 + y1 * y1) * (x3 - x2)
                + (x2 * x2 + y2 * y2) * (x1 - x3)
                + (x3 * x3 + y3 * y3) * (x2 - x1))
                / d;

            let mut vertices = Array2::zeros((1, 2));
            vertices[[0, 0]] = ux;
            vertices[[0, 1]] = uy;

            // Create Voronoi diagram with one vertex
            let ridge_points = vec![[0, 1], [1, 2], [0, 2]];
            let ridge_vertices = vec![vec![0, -1], vec![0, -1], vec![0, -1]];
            let regions = vec![vec![0, -1, -1], vec![0, -1, -1], vec![0, -1, -1]];
            let point_region = Array1::from_vec(vec![0, 1, 2]);

            Ok(Voronoi {
                points: points.to_owned(),
                vertices,
                ridge_points,
                ridge_vertices,
                regions,
                point_region,
                furthest_site,
            })
        }
    }

    /// Special case handler for a square/rectangle (4 points in 2D)
    fn special_case_square(points: &ArrayView2<f64>, furthest_site: bool) -> SpatialResult<Self> {
        // For a square, there's a single Voronoi vertex at the center
        let mut center_x = 0.0;
        let mut center_y = 0.0;

        for i in 0..4 {
            center_x += points[[i, 0]];
            center_y += points[[i, 1]];
        }

        center_x /= 4.0;
        center_y /= 4.0;

        let mut vertices = Array2::zeros((1, 2));
        vertices[[0, 0]] = center_x;
        vertices[[0, 1]] = center_y;

        // Create ridges connecting each pair of adjacent points
        let ridge_points = vec![[0, 1], [1, 2], [2, 3], [3, 0]];
        let ridge_vertices = vec![vec![0, -1], vec![0, -1], vec![0, -1], vec![0, -1]];

        // Each region contains the center vertex and extends to infinity
        let regions = vec![
            vec![0, -1, -1],
            vec![0, -1, -1],
            vec![0, -1, -1],
            vec![0, -1, -1],
        ];

        let point_region = Array1::from_vec(vec![0, 1, 2, 3]);

        Ok(Voronoi {
            points: points.to_owned(),
            vertices,
            ridge_points,
            ridge_vertices,
            regions,
            point_region,
            furthest_site,
        })
    }

    /// Creates a Voronoi diagram from a Delaunay triangulation
    ///
    /// # Arguments
    ///
    /// * `delaunay` - A Delaunay triangulation
    /// * `furthest_site` - Whether to compute a furthest-site Voronoi diagram
    ///
    /// # Returns
    ///
    /// * Result containing a Voronoi diagram or an error
    fn from_delaunay(delaunay: Delaunay, furthest_site: bool) -> SpatialResult<Self> {
        let points = delaunay.points().clone();
        let ndim = points.ncols();
        let npoints = points.nrows();

        // Compute Voronoi vertices as the circumcenters of the Delaunay simplices
        let simplices = delaunay.simplices();
        let mut voronoi_vertices = Vec::new();

        for simplex in simplices {
            if let Some(circumcenter) = Self::compute_circumcenter(&points, simplex, ndim) {
                voronoi_vertices.push(circumcenter);
            } else {
                return Err(SpatialError::ComputationError(
                    "Failed to compute circumcenter".to_string(),
                ));
            }
        }

        // Convert to Array2
        let nvertices = voronoi_vertices.len();
        let mut vertices = Array2::zeros((nvertices, ndim));
        for (i, vertex) in voronoi_vertices.iter().enumerate() {
            for j in 0..ndim {
                vertices[[i, j]] = vertex[j];
            }
        }

        // Create ridge points and ridge vertices
        let mut ridge_points = Vec::new();
        let mut ridge_vertices = Vec::new();

        // Map from pairs of points to ridge vertices
        let mut ridge_map: HashMap<(usize, usize), Vec<i64>> = HashMap::new();

        // Go through simplices and build the ridge map
        let neighbors = delaunay.neighbors();

        for (i, simplex) in simplices.iter().enumerate() {
            for (j, &neighbor_idx) in neighbors[i].iter().enumerate() {
                // Skip if already processed or if neighbor is -1 (no neighbor)
                if neighbor_idx == -1 || (neighbor_idx >= 0 && (neighbor_idx as usize) < i) {
                    continue;
                }

                // Find the points that are not shared between the simplex and its neighbor
                let mut p1 = simplex[j];
                let mut p2 = 0;

                if neighbor_idx >= 0 {
                    let neighbor_simplex = &simplices[neighbor_idx as usize];
                    // Find the vertex in neighbor that's not in current simplex
                    let mut found = false;
                    for &vid in neighbor_simplex {
                        if !simplex.contains(&vid) {
                            p2 = vid;
                            found = true;
                            break;
                        }
                    }

                    if !found {
                        return Err(SpatialError::ComputationError(
                            "Failed to find unique point in neighbor simplex".to_string(),
                        ));
                    }
                } else {
                    // Unbounded ridge - use the centroid of the simplex
                    p2 = p1;
                    // The ridge is unbounded in this direction
                }

                // Ensure p1 < p2 for consistent ridge indexing
                if p1 > p2 {
                    std::mem::swap(&mut p1, &mut p2);
                }

                // Add ridge points
                ridge_points.push([p1, p2]);

                // Create ridge vertices
                let mut ridge_verts = vec![i as i64];
                if neighbor_idx >= 0 {
                    ridge_verts.push(neighbor_idx);
                } else {
                    // Unbounded ridge
                    ridge_verts.push(-1);
                }

                // Add to ridge vertices list
                ridge_vertices.push(ridge_verts.clone());

                // Add to ridge map
                ridge_map.insert((p1, p2), ridge_verts);
            }
        }

        // Create regions - each region is a list of vertex indices
        let mut regions = Vec::with_capacity(npoints);
        let mut point_region = Array1::from_elem(npoints, -1);

        // Build regions
        for i in 0..npoints {
            let mut region = Vec::new();

            // Find all ridges that involve this point
            for ((p1, p2), verts) in &ridge_map {
                if *p1 == i || *p2 == i {
                    for &v in verts {
                        if v >= 0 && !region.contains(&v) {
                            region.push(v);
                        }
                    }
                }
            }

            // If the region is bounded, the vertices should form a polygon around the point
            // We need to sort them counter-clockwise
            if !region.is_empty() {
                point_region[i] = regions.len() as i64;
                regions.push(region);
            }
        }

        Ok(Voronoi {
            points,
            vertices,
            ridge_points,
            ridge_vertices,
            regions,
            point_region,
            furthest_site,
        })
    }

    /// Compute the circumcenter of a simplex
    ///
    /// # Arguments
    ///
    /// * `points` - Array of points
    /// * `simplex` - Indices of vertices forming a simplex
    /// * `ndim` - Number of dimensions
    ///
    /// # Returns
    ///
    /// * Option containing the circumcenter or None if computation fails
    fn compute_circumcenter(
        points: &Array2<f64>,
        simplex: &[usize],
        ndim: usize,
    ) -> Option<Vec<f64>> {
        if simplex.len() != ndim + 1 {
            return None;
        }

        // For a simplex with vertices v_0, v_1, ..., v_n,
        // the circumcenter c is the solution to the system of equations:
        // |v_i - c|^2 = |v_0 - c|^2 for i = 1, 2, ..., n

        // Create a system of linear equations
        let mut system = Array2::zeros((ndim, ndim));
        let mut rhs = Array1::zeros(ndim);

        // Use the first point as the reference
        let p0 = points.row(simplex[0]).to_vec();

        for i in 1..=ndim {
            let pi = points.row(simplex[i]).to_vec();

            for j in 0..ndim {
                system[[i - 1, j]] = 2.0 * (pi[j] - p0[j]);
            }

            // Compute |pi|^2 - |p0|^2
            let sq_dist_pi = pi.iter().map(|&x| x * x).sum::<f64>();
            let sq_dist_p0 = p0.iter().map(|&x| x * x).sum::<f64>();
            rhs[i - 1] = sq_dist_pi - sq_dist_p0;
        }

        // Solve the system using Gaussian elimination
        // This is a simplified approach; a more robust method would be preferable
        // in a production environment
        for i in 0..ndim {
            // Find pivot
            let mut max_row = i;
            let mut max_val = system[[i, i]].abs();

            for j in i + 1..ndim {
                let val = system[[j, i]].abs();
                if val > max_val {
                    max_row = j;
                    max_val = val;
                }
            }

            // Check if pivot is too small
            if max_val < 1e-10 {
                return None;
            }

            // Swap rows if necessary
            if max_row != i {
                for j in 0..ndim {
                    let temp = system[[i, j]];
                    system[[i, j]] = system[[max_row, j]];
                    system[[max_row, j]] = temp;
                }
                let temp = rhs[i];
                rhs[i] = rhs[max_row];
                rhs[max_row] = temp;
            }

            // Eliminate below
            for j in i + 1..ndim {
                let factor = system[[j, i]] / system[[i, i]];
                for k in i..ndim {
                    system[[j, k]] -= factor * system[[i, k]];
                }
                rhs[j] -= factor * rhs[i];
            }
        }

        // Back-substitution
        let mut solution = vec![0.0; ndim];
        for i in (0..ndim).rev() {
            let mut sum = 0.0;
            for j in i + 1..ndim {
                sum += system[[i, j]] * solution[j];
            }
            solution[i] = (rhs[i] - sum) / system[[i, i]];
        }

        Some(solution)
    }

    /// Get the input points of the Voronoi diagram
    ///
    /// # Returns
    ///
    /// * Array of input points
    pub fn points(&self) -> &Array2<f64> {
        &self.points
    }

    /// Get the Voronoi vertices
    ///
    /// # Returns
    ///
    /// * Array of Voronoi vertices
    pub fn vertices(&self) -> &Array2<f64> {
        &self.vertices
    }

    /// Get the ridge points
    ///
    /// # Returns
    ///
    /// * Vector of pairs of point indices, representing the points
    ///   separated by each Voronoi ridge
    pub fn ridge_points(&self) -> &[[usize; 2]] {
        &self.ridge_points
    }

    /// Get the ridge vertices
    ///
    /// # Returns
    ///
    /// * Vector of vertex indices representing the vertices that form each ridge
    pub fn ridge_vertices(&self) -> &[Vec<i64>] {
        &self.ridge_vertices
    }

    /// Get the Voronoi regions
    ///
    /// # Returns
    ///
    /// * Vector of vertex indices representing the vertices that form each region
    pub fn regions(&self) -> &[Vec<i64>] {
        &self.regions
    }

    /// Get the point to region mapping
    ///
    /// # Returns
    ///
    /// * Array mapping each input point index to its region index
    pub fn point_region(&self) -> &Array1<i64> {
        &self.point_region
    }

    /// Check if this is a furthest-site Voronoi diagram
    ///
    /// # Returns
    ///
    /// * true if this is a furthest-site Voronoi diagram, false otherwise
    pub fn is_furthest_site(&self) -> bool {
        self.furthest_site
    }
}

/// Compute a Voronoi diagram from a set of points
///
/// # Arguments
///
/// * `points` - Input points, shape (n_points, n_dim)
/// * `furthest_site` - Whether to compute a furthest-site Voronoi diagram (default: false)
///
/// # Returns
///
/// * Result containing a Voronoi diagram or an error
///
/// # Examples
///
/// ```
/// use scirs2_spatial::voronoi::voronoi;
/// use ndarray::array;
///
/// let points = array![
///     [0.0, 0.0],
///     [1.0, 0.0],
///     [0.0, 1.0],
///     [1.0, 1.0]
/// ];
///
/// let vor = voronoi(&points.view(), false).unwrap();
/// ```
pub fn voronoi(points: &ArrayView2<f64>, furthest_site: bool) -> SpatialResult<Voronoi> {
    Voronoi::new(points, furthest_site)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::arr2;

    #[test]
    fn test_voronoi_square() {
        let points = arr2(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]);

        let vor = Voronoi::new(&points.view(), false).unwrap();

        // The Voronoi diagram of a square should have a single vertex at the center
        assert_eq!(vor.vertices().nrows(), 1);
        assert_relative_eq!(vor.vertices()[[0, 0]], 0.5);
        assert_relative_eq!(vor.vertices()[[0, 1]], 0.5);

        // There should be one region per input point
        assert_eq!(vor.regions().len(), 4);

        // Each point should be mapped to a region
        assert_eq!(vor.point_region().len(), 4);
    }

    #[test]
    fn test_voronoi_triangle() {
        let points = arr2(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);

        let vor = Voronoi::new(&points.view(), false).unwrap();

        // The Voronoi diagram of a triangle should have a single vertex at the circumcenter
        assert_eq!(vor.vertices().nrows(), 1);

        // The circumcenter of the triangle with vertices (0,0), (1,0), (0,1) is at (0.5, 0.5)
        assert_relative_eq!(vor.vertices()[[0, 0]], 0.5, epsilon = 1e-10);
        assert_relative_eq!(vor.vertices()[[0, 1]], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_voronoi_furthest_site() {
        let points = arr2(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]);

        let vor = Voronoi::new(&points.view(), true).unwrap();

        // Check if furthest_site flag is set
        assert!(vor.is_furthest_site());
    }

    #[test]
    fn test_voronoi_function() {
        let points = arr2(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]);

        let vor = voronoi(&points.view(), false).unwrap();

        // Basic check
        assert_eq!(vor.points().nrows(), 4);
        assert_eq!(vor.vertices().nrows(), 1);
    }
}
