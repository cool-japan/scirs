//! Generic implementations of spatial algorithms
//!
//! This module provides generic implementations of common spatial algorithms
//! that can work with different numeric types and point representations.
//! These algorithms leverage the traits defined in the `generic_traits` module
//! to provide flexible, reusable implementations.
//!
//! # Features
//!
//! - **Generic KD-Tree**: Works with any SpatialPoint implementation
//! - **Generic distance calculations**: Support for different metrics and types
//! - **Generic convex hull**: Templated hull algorithms
//! - **Generic clustering**: K-means and other clustering algorithms
//! - **Type safety**: Compile-time dimension and type checking where possible
//!
//! # Examples
//!
//! ```
//! use scirs2_spatial::generic_algorithms::{GenericKDTree, GenericDistanceMatrix};
//! use scirs2_spatial::generic_traits::{Point, EuclideanMetric};
//!
//! // Create points with different numeric types
//! let points_f32 = vec![
//!     Point::new_2d(1.0f32, 2.0f32),
//!     Point::new_2d(3.0f32, 4.0f32),
//! ];
//!
//! let points_f64 = vec![
//!     Point::new_2d(1.0f64, 2.0f64),
//!     Point::new_2d(3.0f64, 4.0f64),
//! ];
//!
//! // Both work with the same algorithm
//! let kdtree_f32 = GenericKDTree::new(&points_f32);
//! let kdtree_f64 = GenericKDTree::new(&points_f64);
//! ```

use crate::generic_traits::{
    DistanceMetric, EuclideanMetric, Point, SpatialArray, SpatialPoint, SpatialScalar,
};
use crate::error::{SpatialError, SpatialResult};
use num_traits::{Zero, One};
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use std::marker::PhantomData;

/// Generic KD-Tree implementation
///
/// This KD-Tree can work with any type that implements SpatialPoint,
/// allowing for flexible point representations and numeric types.
#[derive(Debug, Clone)]
pub struct GenericKDTree<T: SpatialScalar, P: SpatialPoint<T>> {
    root: Option<Box<KDNode<T, P>>>,
    points: Vec<P>,
    dimension: usize,
}

#[derive(Debug, Clone)]
struct KDNode<T: SpatialScalar, P: SpatialPoint<T>> {
    point_index: usize,
    splitting_dimension: usize,
    left: Option<Box<KDNode<T, P>>>,
    right: Option<Box<KDNode<T, P>>>,
    _phantom: PhantomData<(T, P)>,
}

impl<T: SpatialScalar, P: SpatialPoint<T> + Clone> GenericKDTree<T, P> {
    /// Create a new KD-Tree from a collection of points
    pub fn new(points: &[P]) -> SpatialResult<Self> {
        if points.is_empty() {
            return Ok(Self {
                root: None,
                points: Vec::new(),
                dimension: 0,
            });
        }
        
        let dimension = points[0].dimension();
        if dimension == 0 {
            return Err(SpatialError::ValueError(
                "Points must have at least one dimension".to_string(),
            ));
        }
        
        // Verify all points have the same dimension
        for point in points {
            if point.dimension() != dimension {
                return Err(SpatialError::ValueError(
                    "All points must have the same dimension".to_string(),
                ));
            }
        }
        
        let points = points.to_vec();
        let mut indices: Vec<usize> = (0..points.len()).collect();
        
        let root = Self::build_tree(&points, &mut indices, 0, dimension);
        
        Ok(Self {
            root,
            points,
            dimension,
        })
    }
    
    /// Build the KD-Tree recursively
    fn build_tree(
        points: &[P],
        indices: &mut [usize],
        depth: usize,
        dimension: usize,
    ) -> Option<Box<KDNode<T, P>>> {
        if indices.is_empty() {
            return None;
        }
        
        let splitting_dimension = depth % dimension;
        
        // Sort indices by the splitting dimension
        indices.sort_by(|&a, &b| {
            let coord_a = points[a].coordinate(splitting_dimension).unwrap_or(T::zero());
            let coord_b = points[b].coordinate(splitting_dimension).unwrap_or(T::zero());
            coord_a.partial_cmp(&coord_b).unwrap_or(Ordering::Equal)
        });
        
        let median = indices.len() / 2;
        let point_index = indices[median];
        
        let (left_indices, right_indices) = indices.split_at_mut(median);
        let right_indices = &mut right_indices[1..]; // Exclude the median
        
        let left = Self::build_tree(points, left_indices, depth + 1, dimension);
        let right = Self::build_tree(points, right_indices, depth + 1, dimension);
        
        Some(Box::new(KDNode {
            point_index,
            splitting_dimension,
            left,
            right,
            _phantom: PhantomData,
        }))
    }
    
    /// Find the k nearest neighbors to a query point
    pub fn k_nearest_neighbors(
        &self,
        query: &P,
        k: usize,
        metric: &dyn DistanceMetric<T, P>,
    ) -> SpatialResult<Vec<(usize, T)>> {
        if k == 0 {
            return Ok(Vec::new());
        }
        
        if query.dimension() != self.dimension {
            return Err(SpatialError::ValueError(
                "Query point dimension must match tree dimension".to_string(),
            ));
        }
        
        let mut heap = BinaryHeap::new();
        
        if let Some(ref root) = self.root {
            self.search_knn(root, query, k, &mut heap, metric);
        }
        
        let mut result: Vec<(usize, T)> = heap.into_sorted_vec()
            .into_iter()
            .map(|item| (item.index, item.distance))
            .collect();
        
        result.reverse(); // BinaryHeap is max-heap, we want min distances first
        Ok(result)
    }
    
    /// Search for k nearest neighbors recursively
    fn search_knn(
        &self,
        node: &KDNode<T, P>,
        query: &P,
        k: usize,
        heap: &mut BinaryHeap<KNNItem<T>>,
        metric: &dyn DistanceMetric<T, P>,
    ) {
        let point = &self.points[node.point_index];
        let distance = metric.distance(query, point);
        
        if heap.len() < k {
            heap.push(KNNItem {
                distance,
                index: node.point_index,
            });
        } else if let Some(top) = heap.peek() {
            if distance < top.distance {
                heap.pop();
                heap.push(KNNItem {
                    distance,
                    index: node.point_index,
                });
            }
        }
        
        // Determine which child to visit first
        let query_coord = query.coordinate(node.splitting_dimension).unwrap_or(T::zero());
        let point_coord = point.coordinate(node.splitting_dimension).unwrap_or(T::zero());
        
        let (first_child, second_child) = if query_coord < point_coord {
            (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };
        
        // Search the closer child first
        if let Some(ref child) = first_child {
            self.search_knn(child, query, k, heap, metric);
        }
        
        // Check if we need to search the other child
        let dimension_distance = (query_coord - point_coord).abs();
        let should_search_other = heap.len() < k || 
            heap.peek().map_or(true, |top| dimension_distance < top.distance);
        
        if should_search_other {
            if let Some(ref child) = second_child {
                self.search_knn(child, query, k, heap, metric);
            }
        }
    }
    
    /// Find all points within a given radius of the query point
    pub fn radius_search(
        &self,
        query: &P,
        radius: T,
        metric: &dyn DistanceMetric<T, P>,
    ) -> SpatialResult<Vec<(usize, T)>> {
        if query.dimension() != self.dimension {
            return Err(SpatialError::ValueError(
                "Query point dimension must match tree dimension".to_string(),
            ));
        }
        
        let mut result = Vec::new();
        
        if let Some(ref root) = self.root {
            self.search_radius(root, query, radius, &mut result, metric);
        }
        
        Ok(result)
    }
    
    /// Search for points within radius recursively
    fn search_radius(
        &self,
        node: &KDNode<T, P>,
        query: &P,
        radius: T,
        result: &mut Vec<(usize, T)>,
        metric: &dyn DistanceMetric<T, P>,
    ) {
        let point = &self.points[node.point_index];
        let distance = metric.distance(query, point);
        
        if distance <= radius {
            result.push((node.point_index, distance));
        }
        
        let query_coord = query.coordinate(node.splitting_dimension).unwrap_or(T::zero());
        let point_coord = point.coordinate(node.splitting_dimension).unwrap_or(T::zero());
        let dimension_distance = (query_coord - point_coord).abs();
        
        // Search left child
        if let Some(ref left) = node.left {
            if query_coord - radius <= point_coord {
                self.search_radius(left, query, radius, result, metric);
            }
        }
        
        // Search right child
        if let Some(ref right) = node.right {
            if query_coord + radius >= point_coord {
                self.search_radius(right, query, radius, result, metric);
            }
        }
    }
}

/// Helper struct for k-nearest neighbor search
#[derive(Debug, Clone)]
struct KNNItem<T: SpatialScalar> {
    distance: T,
    index: usize,
}

impl<T: SpatialScalar> PartialEq for KNNItem<T> {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl<T: SpatialScalar> Eq for KNNItem<T> {}

impl<T: SpatialScalar> PartialOrd for KNNItem<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl<T: SpatialScalar> Ord for KNNItem<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Generic distance matrix computation
pub struct GenericDistanceMatrix;

impl GenericDistanceMatrix {
    /// Compute pairwise distance matrix between points
    pub fn compute<T, P, M>(
        points: &[P],
        metric: &M,
    ) -> SpatialResult<Vec<Vec<T>>>
    where
        T: SpatialScalar,
        P: SpatialPoint<T>,
        M: DistanceMetric<T, P>,
    {
        let n = points.len();
        let mut matrix = vec![vec![T::zero(); n]; n];
        
        for i in 0..n {
            for j in i..n {
                let distance = if i == j {
                    T::zero()
                } else {
                    metric.distance(&points[i], &points[j])
                };
                
                matrix[i][j] = distance;
                matrix[j][i] = distance;
            }
        }
        
        Ok(matrix)
    }
    
    /// Compute condensed distance matrix (upper triangle only)
    pub fn compute_condensed<T, P, M>(
        points: &[P],
        metric: &M,
    ) -> SpatialResult<Vec<T>>
    where
        T: SpatialScalar,
        P: SpatialPoint<T>,
        M: DistanceMetric<T, P>,
    {
        let n = points.len();
        let mut distances = Vec::with_capacity(n * (n - 1) / 2);
        
        for i in 0..n {
            for j in (i + 1)..n {
                distances.push(metric.distance(&points[i], &points[j]));
            }
        }
        
        Ok(distances)
    }
}

/// Generic K-means clustering implementation
pub struct GenericKMeans<T: SpatialScalar, P: SpatialPoint<T>> {
    k: usize,
    max_iterations: usize,
    tolerance: T,
    _phantom: PhantomData<(T, P)>,
}

impl<T: SpatialScalar, P: SpatialPoint<T> + Clone> GenericKMeans<T, P> {
    /// Create a new K-means clusterer
    pub fn new(k: usize) -> Self {
        Self {
            k,
            max_iterations: 100,
            tolerance: T::from_f64(1e-6).unwrap_or(T::epsilon()),
            _phantom: PhantomData,
        }
    }
    
    /// Set the maximum number of iterations
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }
    
    /// Set the convergence tolerance
    pub fn with_tolerance(mut self, tolerance: T) -> Self {
        self.tolerance = tolerance;
        self
    }
    
    /// Perform K-means clustering
    pub fn fit(&self, points: &[P]) -> SpatialResult<KMeansResult<T, P>> {
        if points.is_empty() {
            return Err(SpatialError::ValueError(
                "Cannot cluster empty point set".to_string(),
            ));
        }
        
        if self.k > points.len() {
            return Err(SpatialError::ValueError(
                "k cannot be larger than the number of points".to_string(),
            ));
        }
        
        let dimension = points[0].dimension();
        
        // Initialize centroids randomly (simple initialization)
        let mut centroids = self.initialize_centroids(points, dimension)?;
        let mut assignments = vec![0; points.len()];
        
        for iteration in 0..self.max_iterations {
            let mut changed = false;
            
            // Assign points to nearest centroids
            for (i, point) in points.iter().enumerate() {
                let mut best_cluster = 0;
                let mut best_distance = T::max_finite();
                
                for (j, centroid) in centroids.iter().enumerate() {
                    let distance = point.distance_to(centroid);
                    if distance < best_distance {
                        best_distance = distance;
                        best_cluster = j;
                    }
                }
                
                if assignments[i] != best_cluster {
                    assignments[i] = best_cluster;
                    changed = true;
                }
            }
            
            // Update centroids
            let old_centroids = centroids.clone();
            centroids = self.update_centroids(points, &assignments, dimension)?;
            
            // Check for convergence
            let max_movement = old_centroids
                .iter()
                .zip(centroids.iter())
                .map(|(old, new)| old.distance_to(new))
                .fold(T::zero(), |acc, dist| if dist > acc { dist } else { acc });
            
            if !changed || max_movement < self.tolerance {
                return Ok(KMeansResult {
                    centroids,
                    assignments,
                    iterations: iteration + 1,
                    converged: max_movement < self.tolerance,
                });
            }
        }
        
        Ok(KMeansResult {
            centroids,
            assignments,
            iterations: self.max_iterations,
            converged: false,
        })
    }
    
    /// Initialize centroids using k-means++
    fn initialize_centroids(&self, points: &[P], dimension: usize) -> SpatialResult<Vec<Point<T>>> {
        let mut centroids = Vec::with_capacity(self.k);
        
        // Choose first centroid randomly
        centroids.push(self.point_to_generic(&points[0]));
        
        // Choose remaining centroids using k-means++ initialization
        for _ in 1..self.k {
            let mut distances = Vec::with_capacity(points.len());
            
            for point in points {
                let min_distance = centroids
                    .iter()
                    .map(|centroid| point.distance_to(centroid))
                    .fold(T::max_finite(), |acc, dist| if dist < acc { dist } else { acc });
                distances.push(min_distance);
            }
            
            // Find the point with maximum distance to nearest centroid
            let max_distance_idx = distances
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            
            centroids.push(self.point_to_generic(&points[max_distance_idx]));
        }
        
        Ok(centroids)
    }
    
    /// Update centroids based on current assignments
    fn update_centroids(
        &self,
        points: &[P],
        assignments: &[usize],
        dimension: usize,
    ) -> SpatialResult<Vec<Point<T>>> {
        let mut centroids = vec![Point::zeros(dimension); self.k];
        let mut counts = vec![0; self.k];
        
        // Sum points for each cluster
        for (point, &cluster) in points.iter().zip(assignments.iter()) {
            for d in 0..dimension {
                if let Some(coord) = point.coordinate(d) {
                    if let Some(centroid_coord) = centroids[cluster].coords_mut().get_mut(d) {
                        *centroid_coord = *centroid_coord + coord;
                    }
                }
            }
            counts[cluster] += 1;
        }
        
        // Average to get centroids
        for (centroid, count) in centroids.iter_mut().zip(counts.iter()) {
            if *count > 0 {
                let count_scalar = T::from(*count).unwrap_or(T::one());
                for coord in centroid.coords_mut() {
                    *coord = *coord / count_scalar;
                }
            }
        }
        
        Ok(centroids)
    }
    
    /// Convert a point to generic Point type
    fn point_to_generic(&self, point: &P) -> Point<T> {
        let coords: Vec<T> = (0..point.dimension())
            .map(|i| point.coordinate(i).unwrap_or(T::zero()))
            .collect();
        Point::new(coords)
    }
}

/// Result of K-means clustering
#[derive(Debug, Clone)]
pub struct KMeansResult<T: SpatialScalar, P: SpatialPoint<T>> {
    /// Final centroids
    pub centroids: Vec<Point<T>>,
    /// Cluster assignment for each point
    pub assignments: Vec<usize>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Whether the algorithm converged
    pub converged: bool,
    
    _phantom: PhantomData<P>,
}

/// Generic convex hull computation using Graham scan
pub struct GenericConvexHull;

impl GenericConvexHull {
    /// Compute 2D convex hull using Graham scan
    pub fn graham_scan_2d<T, P>(points: &[P]) -> SpatialResult<Vec<Point<T>>>
    where
        T: SpatialScalar,
        P: SpatialPoint<T> + Clone,
    {
        if points.is_empty() {
            return Ok(Vec::new());
        }
        
        if points.len() < 3 {
            return Ok(points.iter().map(|p| Self::to_generic_point(p)).collect());
        }
        
        // Verify all points are 2D
        for point in points {
            if point.dimension() != 2 {
                return Err(SpatialError::ValueError(
                    "All points must be 2D for 2D convex hull".to_string(),
                ));
            }
        }
        
        let mut generic_points: Vec<Point<T>> = points.iter().map(|p| Self::to_generic_point(p)).collect();
        
        // Find the point with lowest y-coordinate (and leftmost if tie)
        let start_idx = generic_points
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let y_cmp = a.coordinate(1).partial_cmp(&b.coordinate(1)).unwrap();
                if y_cmp == Ordering::Equal {
                    a.coordinate(0).partial_cmp(&b.coordinate(0)).unwrap()
                } else {
                    y_cmp
                }
            })
            .map(|(idx, _)| idx)
            .unwrap();
        
        generic_points.swap(0, start_idx);
        let start_point = generic_points[0].clone();
        
        // Sort points by polar angle with respect to start point
        generic_points[1..].sort_by(|a, b| {
            let angle_a = Self::polar_angle(&start_point, a);
            let angle_b = Self::polar_angle(&start_point, b);
            angle_a.partial_cmp(&angle_b).unwrap_or(Ordering::Equal)
        });
        
        // Build convex hull
        let mut hull = Vec::new();
        for point in generic_points {
            while hull.len() > 1 && Self::cross_product(&hull[hull.len()-2], &hull[hull.len()-1], &point) <= T::zero() {
                hull.pop();
            }
            hull.push(point);
        }
        
        Ok(hull)
    }
    
    /// Convert a point to generic Point type
    fn to_generic_point<T, P>(point: &P) -> Point<T>
    where
        T: SpatialScalar,
        P: SpatialPoint<T>,
    {
        let coords: Vec<T> = (0..point.dimension())
            .map(|i| point.coordinate(i).unwrap_or(T::zero()))
            .collect();
        Point::new(coords)
    }
    
    /// Calculate polar angle from start to point
    fn polar_angle<T: SpatialScalar>(start: &Point<T>, point: &Point<T>) -> T {
        let dx = point.coordinate(0).unwrap_or(T::zero()) - start.coordinate(0).unwrap_or(T::zero());
        let dy = point.coordinate(1).unwrap_or(T::zero()) - start.coordinate(1).unwrap_or(T::zero());
        dy.atan2(dx)
    }
    
    /// Calculate cross product for 2D points
    fn cross_product<T: SpatialScalar>(a: &Point<T>, b: &Point<T>, c: &Point<T>) -> T {
        let ab_x = b.coordinate(0).unwrap_or(T::zero()) - a.coordinate(0).unwrap_or(T::zero());
        let ab_y = b.coordinate(1).unwrap_or(T::zero()) - a.coordinate(1).unwrap_or(T::zero());
        let ac_x = c.coordinate(0).unwrap_or(T::zero()) - a.coordinate(0).unwrap_or(T::zero());
        let ac_y = c.coordinate(1).unwrap_or(T::zero()) - a.coordinate(1).unwrap_or(T::zero());
        
        ab_x * ac_y - ab_y * ac_x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generic_traits::{EuclideanMetric, ManhattanMetric, Point};
    use approx::assert_relative_eq;
    
    #[test]
    fn test_generic_kdtree() {
        let points = vec![
            Point::new_2d(0.0f64, 0.0),
            Point::new_2d(1.0, 0.0),
            Point::new_2d(0.0, 1.0),
            Point::new_2d(1.0, 1.0),
            Point::new_2d(0.5, 0.5),
        ];
        
        let kdtree = GenericKDTree::new(&points).unwrap();
        let euclidean = EuclideanMetric;
        
        let query = Point::new_2d(0.6, 0.6);
        let neighbors = kdtree.k_nearest_neighbors(&query, 2, &euclidean).unwrap();
        
        assert_eq!(neighbors.len(), 2);
        // Should find the center point (0.5, 0.5) as closest
        assert_eq!(neighbors[0].0, 4);
    }
    
    #[test]
    fn test_generic_distance_matrix() {
        let points = vec![
            Point::new_2d(0.0f32, 0.0f32),
            Point::new_2d(1.0, 0.0),
            Point::new_2d(0.0, 1.0),
        ];
        
        let euclidean = EuclideanMetric;
        let matrix = GenericDistanceMatrix::compute(&points, &euclidean).unwrap();
        
        assert_eq!(matrix.len(), 3);
        assert_eq!(matrix[0].len(), 3);
        assert_relative_eq!(matrix[0][0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(matrix[0][1], 1.0, epsilon = 1e-6);
        assert_relative_eq!(matrix[0][2], 1.0, epsilon = 1e-6);
        assert_relative_eq!(matrix[1][2], 2.0f32.sqrt(), epsilon = 1e-6);
    }
    
    #[test]
    fn test_generic_kmeans() {
        let points = vec![
            Point::new_2d(0.0f64, 0.0),
            Point::new_2d(0.1, 0.1),
            Point::new_2d(5.0, 5.0),
            Point::new_2d(5.1, 5.1),
        ];
        
        let kmeans = GenericKMeans::new(2);
        let result = kmeans.fit(&points).unwrap();
        
        assert_eq!(result.centroids.len(), 2);
        assert_eq!(result.assignments.len(), 4);
        
        // Points should be clustered into two groups
        assert_eq!(result.assignments[0], result.assignments[1]);
        assert_eq!(result.assignments[2], result.assignments[3]);
        assert_ne!(result.assignments[0], result.assignments[2]);
    }
    
    #[test]
    fn test_generic_convex_hull() {
        let points = vec![
            Point::new_2d(0.0f64, 0.0),
            Point::new_2d(1.0, 0.0),
            Point::new_2d(1.0, 1.0),
            Point::new_2d(0.0, 1.0),
            Point::new_2d(0.5, 0.5), // Interior point
        ];
        
        let hull = GenericConvexHull::graham_scan_2d(&points).unwrap();
        
        // Should have 4 points (the square corners), interior point excluded
        assert_eq!(hull.len(), 4);
    }
    
    #[test]
    fn test_different_numeric_types() {
        // Test with f32
        let points_f32 = vec![
            Point::new_2d(0.0f32, 0.0f32),
            Point::new_2d(1.0f32, 1.0f32),
        ];
        
        let kdtree_f32 = GenericKDTree::new(&points_f32).unwrap();
        let euclidean = EuclideanMetric;
        let query_f32 = Point::new_2d(0.5f32, 0.5f32);
        let neighbors_f32 = kdtree_f32.k_nearest_neighbors(&query_f32, 1, &euclidean).unwrap();
        
        assert_eq!(neighbors_f32.len(), 1);
        
        // Test with f64
        let points_f64 = vec![
            Point::new_2d(0.0f64, 0.0f64),
            Point::new_2d(1.0f64, 1.0f64),
        ];
        
        let kdtree_f64 = GenericKDTree::new(&points_f64).unwrap();
        let query_f64 = Point::new_2d(0.5f64, 0.5f64);
        let neighbors_f64 = kdtree_f64.k_nearest_neighbors(&query_f64, 1, &euclidean).unwrap();
        
        assert_eq!(neighbors_f64.len(), 1);
    }
}