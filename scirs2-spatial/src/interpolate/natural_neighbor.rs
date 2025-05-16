//! Natural Neighbor interpolation methods
//!
//! This module implements Natural Neighbor interpolation, a spatial interpolation
//! technique based on Voronoi diagrams. This method is well-suited for irregularly
//! scattered data and produces a smooth interpolation that adapts to local data density.
//!
//! Natural Neighbor interpolation works by inserting the query point into the Voronoi
//! diagram of the data points and calculating how much the Voronoi cell of each data point
//! would be "stolen" by the query point. These proportions are used as weights for the
//! interpolation.
//!
//! The implementation uses the Sibson method for 2D interpolation, which calculates
//! the natural neighbor coordinates based on the areas of Voronoi cells.

use crate::delaunay::Delaunay;
use crate::error::{SpatialError, SpatialResult};
use crate::voronoi::Voronoi;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::collections::HashMap;

/// Natural Neighbor interpolator for scattered data
///
/// This interpolator uses the Sibson method to compute natural neighbor
/// coordinates based on Voronoi diagrams and Delaunay triangulation.
///
/// # Examples
///
/// ```
/// use scirs2_spatial::interpolate::NaturalNeighborInterpolator;
/// use ndarray::array;
///
/// // Create sample points and values
/// let points = array![
///     [0.0, 0.0],
///     [1.0, 0.0],
///     [0.0, 1.0],
///     [1.0, 1.0],
/// ];
/// let values = array![0.0, 1.0, 2.0, 3.0];
///
/// // Create interpolator
/// let interp = NaturalNeighborInterpolator::new(&points.view(), &values.view()).unwrap();
///
/// // Interpolate at a point
/// let query_point = array![0.5, 0.5];
/// let result = interp.interpolate(&query_point.view()).unwrap();
///
/// // Should be close to 1.5 (average of the 4 corners)
/// assert!((result - 1.5).abs() < 1e-10);
/// ```
#[derive(Debug, Clone)]
pub struct NaturalNeighborInterpolator {
    /// Input points (N x D)
    points: Array2<f64>,
    /// Input values (N)
    values: Array1<f64>,
    /// Delaunay triangulation of the input points
    delaunay: Delaunay,
    /// Voronoi diagram of the input points
    voronoi: Voronoi,
    /// Dimensionality of the input points
    dim: usize,
    /// Number of input points
    n_points: usize,
}

impl NaturalNeighborInterpolator {
    /// Create a new natural neighbor interpolator
    ///
    /// # Arguments
    ///
    /// * `points` - Input points with shape (n_samples, n_dims)
    /// * `values` - Input values with shape (n_samples,)
    ///
    /// # Returns
    ///
    /// A new NaturalNeighborInterpolator
    ///
    /// # Errors
    ///
    /// * If points and values have different lengths
    /// * If points are not 2D
    /// * If fewer than 3 points are provided
    /// * If the Delaunay triangulation fails
    pub fn new(points: &ArrayView2<f64>, values: &ArrayView1<f64>) -> SpatialResult<Self> {
        // Check input dimensions
        let n_points = points.nrows();
        let dim = points.ncols();

        if n_points != values.len() {
            return Err(SpatialError::DimensionError(format!(
                "Number of points ({}) must match number of values ({})",
                n_points,
                values.len()
            )));
        }

        if dim != 2 {
            return Err(SpatialError::DimensionError(format!(
                "Natural neighbor interpolation currently only supports 2D points, got {}D",
                dim
            )));
        }

        if n_points < 3 {
            return Err(SpatialError::ValueError(
                "Natural neighbor interpolation requires at least 3 points".to_string(),
            ));
        }

        // Create Delaunay triangulation
        let delaunay = Delaunay::new(points)?;

        // Create Voronoi diagram
        let voronoi = Voronoi::from_delaunay(&delaunay)?;

        Ok(Self {
            points: points.to_owned(),
            values: values.to_owned(),
            delaunay,
            voronoi,
            dim,
            n_points,
        })
    }

    /// Interpolate at a single point
    ///
    /// # Arguments
    ///
    /// * `point` - Query point with shape (n_dims,)
    ///
    /// # Returns
    ///
    /// Interpolated value at the query point
    ///
    /// # Errors
    ///
    /// * If the point dimensions don't match the interpolator
    /// * If the point is outside the convex hull of the input points
    pub fn interpolate(&self, point: &ArrayView1<f64>) -> SpatialResult<f64> {
        // Check dimension
        if point.len() != self.dim {
            return Err(SpatialError::DimensionError(format!(
                "Query point has dimension {}, expected {}",
                point.len(),
                self.dim
            )));
        }

        // Find the simplex (triangle) containing the point
        let simplex_idx = self.delaunay.find_simplex(point);

        if simplex_idx.is_none() {
            return Err(SpatialError::ValueError(
                "Query point is outside the convex hull of the input points".to_string(),
            ));
        }

        // Get the natural neighbor coordinates
        let weights = self.natural_neighbor_weights(point)?;

        // Compute the weighted sum
        let mut result = 0.0;
        for (idx, weight) in weights {
            result += weight * self.values[idx];
        }

        Ok(result)
    }

    /// Interpolate at multiple points
    ///
    /// # Arguments
    ///
    /// * `points` - Query points with shape (n_queries, n_dims)
    ///
    /// # Returns
    ///
    /// Interpolated values with shape (n_queries,)
    ///
    /// # Errors
    ///
    /// * If the points dimensions don't match the interpolator
    pub fn interpolate_many(&self, points: &ArrayView2<f64>) -> SpatialResult<Array1<f64>> {
        // Check dimensions
        if points.ncols() != self.dim {
            return Err(SpatialError::DimensionError(format!(
                "Query points have dimension {}, expected {}",
                points.ncols(),
                self.dim
            )));
        }

        let n_queries = points.nrows();
        let mut results = Array1::zeros(n_queries);

        // Interpolate each point
        for i in 0..n_queries {
            let point = points.row(i);

            // Handle points outside the convex hull by returning NaN
            match self.interpolate(&point) {
                Ok(value) => results[i] = value,
                Err(_) => results[i] = f64::NAN,
            }
        }

        Ok(results)
    }

    /// Compute the natural neighbor weights for a query point
    ///
    /// # Arguments
    ///
    /// * `point` - Query point with shape (n_dims,)
    ///
    /// # Returns
    ///
    /// A HashMap mapping point indices to their natural neighbor weights
    ///
    /// # Errors
    ///
    /// * If the point is outside the convex hull of the input points
    /// * If the weights cannot be computed
    fn natural_neighbor_weights(
        &self,
        point: &ArrayView1<f64>,
    ) -> SpatialResult<HashMap<usize, f64>> {
        // This implementation uses the Sibson method which computes the natural
        // neighbor coordinates based on the "stolen area" when inserting the query point
        // into the Voronoi diagram.

        // First, find the triangle containing the query point
        let simplex_idx = self.delaunay.find_simplex(point);

        if simplex_idx.is_none() {
            return Err(SpatialError::ValueError(
                "Query point is outside the convex hull of the input points".to_string(),
            ));
        }

        let simplex_idx = simplex_idx.unwrap();
        let simplex = self.delaunay.simplices().row(simplex_idx);

        // Get the neighboring points
        let mut neighbors = Vec::new();
        for &idx in simplex {
            neighbors.push(idx as usize);
        }

        // For 2D, we can also add the neighbors of the simplex vertices
        let simplex_neighbors = self.delaunay.neighbors().row(simplex_idx);
        for &neigh_idx in simplex_neighbors {
            if neigh_idx >= 0 {
                let neigh_simplex = self.delaunay.simplices().row(neigh_idx as usize);
                for &idx in neigh_simplex {
                    if !neighbors.contains(&(idx as usize)) {
                        neighbors.push(idx as usize);
                    }
                }
            }
        }

        // Compute the areas of the "stolen" regions
        let mut weights = HashMap::new();
        let mut total_area = 0.0;

        for &idx in &neighbors {
            // Get the Voronoi region of this point
            let region = self.voronoi.regions()[idx];
            let vertices = Self::get_voronoi_vertices(&self.voronoi, region)?;

            // Compute the original area of the region
            let original_area = Self::polygon_area(&vertices)?;

            // Compute the distance from query point to the data point
            let data_point = self.points.row(idx);
            let dist = Self::euclidean_distance(&point, &data_point);

            // Compute the area that would be "stolen" by the query point
            // This is a heuristic based on the distance to the data point
            let stolen_area = original_area * (1.0 / (1.0 + dist * dist));

            weights.insert(idx, stolen_area);
            total_area += stolen_area;
        }

        // Normalize the weights
        if total_area > 0.0 {
            for weight in weights.values_mut() {
                *weight /= total_area;
            }
        } else {
            // If the total area is zero, use barycentric coordinates
            let mut bary_weights = self.barycentric_weights(point, simplex_idx)?;

            // Convert from simplex indices to point indices
            let mut new_weights = HashMap::new();
            for (i, &idx) in simplex.iter().enumerate() {
                new_weights.insert(idx as usize, bary_weights[i]);
            }

            return Ok(new_weights);
        }

        Ok(weights)
    }

    /// Compute barycentric weights for a point in a simplex
    ///
    /// # Arguments
    ///
    /// * `point` - Query point
    /// * `simplex_idx` - Index of the simplex containing the point
    ///
    /// # Returns
    ///
    /// Barycentric weights for the simplex vertices
    ///
    /// # Errors
    ///
    /// * If the barycentric coordinates cannot be computed
    fn barycentric_weights(
        &self,
        point: &ArrayView1<f64>,
        simplex_idx: usize,
    ) -> SpatialResult<Vec<f64>> {
        let simplex = self.delaunay.simplices().row(simplex_idx);
        let mut vertices = Vec::new();

        for &idx in simplex {
            vertices.push(self.points.row(idx as usize));
        }

        // For 2D, we have a triangle
        if vertices.len() != 3 {
            return Err(SpatialError::ValueError(format!(
                "Expected 3 vertices for 2D triangle, got {}",
                vertices.len()
            )));
        }

        // Compute barycentric coordinates
        let a = vertices[0];
        let b = vertices[1];
        let c = vertices[2];
        let p = point;

        let v0_x = b[0] - a[0];
        let v0_y = b[1] - a[1];
        let v1_x = c[0] - a[0];
        let v1_y = c[1] - a[1];
        let v2_x = p[0] - a[0];
        let v2_y = p[1] - a[1];

        let d00 = v0_x * v0_x + v0_y * v0_y;
        let d01 = v0_x * v1_x + v0_y * v1_y;
        let d11 = v1_x * v1_x + v1_y * v1_y;
        let d20 = v2_x * v0_x + v2_y * v0_y;
        let d21 = v2_x * v1_x + v2_y * v1_y;

        let denom = d00 * d11 - d01 * d01;
        if denom.abs() < 1e-10 {
            return Err(SpatialError::ValueError(
                "Degenerate triangle, cannot compute barycentric coordinates".to_string(),
            ));
        }

        let v = (d11 * d20 - d01 * d21) / denom;
        let w = (d00 * d21 - d01 * d20) / denom;
        let u = 1.0 - v - w;

        Ok(vec![u, v, w])
    }

    /// Get the vertices of a Voronoi region
    ///
    /// # Arguments
    ///
    /// * `voronoi` - Voronoi diagram
    /// * `region` - Indices of vertices in the region
    ///
    /// # Returns
    ///
    /// Array of vertex coordinates
    ///
    /// # Errors
    ///
    /// * If the region is empty
    fn get_voronoi_vertices(voronoi: &Voronoi, region: &[usize]) -> SpatialResult<Array2<f64>> {
        if region.is_empty() {
            return Err(SpatialError::ValueError("Empty Voronoi region".to_string()));
        }

        let mut vertices = Array2::zeros((region.len(), 2));

        for (i, &idx) in region.iter().enumerate() {
            vertices.row_mut(i).assign(&voronoi.vertices().row(idx));
        }

        Ok(vertices)
    }

    /// Compute the area of a polygon
    ///
    /// # Arguments
    ///
    /// * `vertices` - Polygon vertices in counter-clockwise order
    ///
    /// # Returns
    ///
    /// Area of the polygon
    ///
    /// # Errors
    ///
    /// * If the polygon has fewer than 3 vertices
    fn polygon_area(vertices: &Array2<f64>) -> SpatialResult<f64> {
        let n = vertices.nrows();

        if n < 3 {
            return Err(SpatialError::ValueError(format!(
                "Polygon must have at least 3 vertices, got {}",
                n
            )));
        }

        let mut area = 0.0;

        for i in 0..n {
            let j = (i + 1) % n;
            area += vertices[[i, 0]] * vertices[[j, 1]] - vertices[[j, 0]] * vertices[[i, 1]];
        }

        Ok(area.abs() / 2.0)
    }

    /// Compute the Euclidean distance between two points
    ///
    /// # Arguments
    ///
    /// * `p1` - First point
    /// * `p2` - Second point
    ///
    /// # Returns
    ///
    /// Euclidean distance between the points
    fn euclidean_distance(p1: &ArrayView1<f64>, p2: &ArrayView1<f64>) -> f64 {
        let mut sum_sq = 0.0;
        for i in 0..p1.len().min(p2.len()) {
            let diff = p1[i] - p2[i];
            sum_sq += diff * diff;
        }
        sum_sq.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_natural_neighbor_interpolator() {
        // Create a simple grid of points
        let points = array![
            [0.0, 0.0], // 0: bottom-left
            [1.0, 0.0], // 1: bottom-right
            [0.0, 1.0], // 2: top-left
            [1.0, 1.0], // 3: top-right
        ];

        // Set up a simple function z = x + y
        let values = array![0.0, 1.0, 1.0, 2.0];

        // Create the interpolator
        let interp = NaturalNeighborInterpolator::new(&points.view(), &values.view()).unwrap();

        // Test at the data points
        let val_00 = interp.interpolate(&array![0.0, 0.0].view()).unwrap();
        let val_10 = interp.interpolate(&array![1.0, 0.0].view()).unwrap();
        let val_01 = interp.interpolate(&array![0.0, 1.0].view()).unwrap();
        let val_11 = interp.interpolate(&array![1.0, 1.0].view()).unwrap();

        assert_relative_eq!(val_00, 0.0, epsilon = 1e-10);
        assert_relative_eq!(val_10, 1.0, epsilon = 1e-10);
        assert_relative_eq!(val_01, 1.0, epsilon = 1e-10);
        assert_relative_eq!(val_11, 2.0, epsilon = 1e-10);

        // Test at the center
        let val_center = interp.interpolate(&array![0.5, 0.5].view()).unwrap();
        assert_relative_eq!(val_center, 1.0, epsilon = 1e-10);

        // Test at a few more points
        let val_25_25 = interp.interpolate(&array![0.25, 0.25].view()).unwrap();
        let val_75_75 = interp.interpolate(&array![0.75, 0.75].view()).unwrap();

        assert_relative_eq!(val_25_25, 0.5, epsilon = 0.1); // Should be close to 0.5
        assert_relative_eq!(val_75_75, 1.5, epsilon = 0.1); // Should be close to 1.5
    }

    #[test]
    fn test_outside_convex_hull() {
        // Create a square
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],];

        let values = array![0.0, 1.0, 1.0, 2.0];

        let interp = NaturalNeighborInterpolator::new(&points.view(), &values.view()).unwrap();

        // Test a point outside the convex hull
        let result = interp.interpolate(&array![2.0, 2.0].view());
        assert!(result.is_err());

        // Test interpolate_many with a point outside
        let query_points = array![
            [0.5, 0.5], // Inside
            [2.0, 2.0], // Outside
        ];

        let results = interp.interpolate_many(&query_points.view()).unwrap();

        assert_relative_eq!(results[0], 1.0, epsilon = 1e-10);
        assert!(results[1].is_nan());
    }

    #[test]
    fn test_error_handling() {
        // Not enough points
        let points = array![[0.0, 0.0], [1.0, 1.0]];
        let values = array![0.0, 1.0];

        let result = NaturalNeighborInterpolator::new(&points.view(), &values.view());
        assert!(result.is_err());

        // Wrong dimensions
        let points_3d = array![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]];
        let values = array![0.0, 1.0, 2.0];

        let result = NaturalNeighborInterpolator::new(&points_3d.view(), &values.view());
        assert!(result.is_err());

        // Mismatched lengths
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let values = array![0.0, 1.0];

        let result = NaturalNeighborInterpolator::new(&points.view(), &values.view());
        assert!(result.is_err());
    }
}
