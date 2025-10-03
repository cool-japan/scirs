//! KD-Tree for efficient nearest neighbor searches
//!
//! This module provides a KD-Tree implementation for efficient
//! nearest neighbor and range searches in multidimensional spaces.
//!
//! The KD-Tree (k-dimensional tree) is a space-partitioning data structure
//! that organizes points in a k-dimensional space. It enables efficient range searches
//! and nearest neighbor searches.
//!
//! # Features
//!
//! * Fast nearest neighbor queries with customizable `k`
//! * Range queries with distance threshold
//! * Support for different distance metrics (Euclidean, Manhattan, Chebyshev, etc.)
//! * Parallel query processing (when using the `parallel` feature)
//! * Customizable leaf size for performance tuning
//!
//! # Examples
//!
//! ```
//! use scirs2_spatial::KDTree;
//! use scirs2_core::ndarray::array;
//!
//! // Create a KD-Tree with points in 2D space
//! let points = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
//! let kdtree = KDTree::new(&points).unwrap();
//!
//! // Find the nearest neighbor to [4.0, 5.0]
//! let (idx, dist) = kdtree.query(&[4.0, 5.0], 1).unwrap();
//! assert_eq!(idx.len(), 1); // Should return exactly one neighbor
//!
//! // Find all points within radius 3.0 of [4.0, 5.0]
//! let (indices, distances) = kdtree.query_radius(&[4.0, 5.0], 3.0).unwrap();
//! ```
//!
//! # Advanced Usage
//!
//! Using custom distance metrics:
//!
//! ```
//! use scirs2_spatial::KDTree;
//! use scirs2_spatial::distance::ManhattanDistance;
//! use scirs2_core::ndarray::array;
//!
//! // Create a KD-Tree with Manhattan distance metric
//! let points = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
//! let metric = ManhattanDistance::new();
//! let kdtree = KDTree::with_metric(&points, metric).unwrap();
//!
//! // Find the nearest neighbor to [4.0, 5.0] using Manhattan distance
//! let (idx, dist) = kdtree.query(&[4.0, 5.0], 1).unwrap();
//! ```
//!
//! Using custom leaf size for performance tuning:
//!
//! ```
//! use scirs2_spatial::KDTree;
//! use scirs2_core::ndarray::array;
//!
//! // Create a KD-Tree with custom leaf size
//! let points = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0],
//!                      [9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]];
//! let leafsize = 2; // Default is 16
//! let kdtree = KDTree::with_leaf_size(&points, leafsize).unwrap();
//! ```

use crate::distance::{Distance, EuclideanDistance};
use crate::error::{SpatialError, SpatialResult};
use crate::safe_conversions::*;
use scirs2_core::ndarray::Array2;
use scirs2_core::numeric::Float;
use std::cmp::Ordering;

// Rayon parallel processing currently not used in this module
#[cfg(feature = "parallel")]
#[allow(unused_imports)]
/// A rectangle representing a hyperrectangle in k-dimensional space
///
/// Used for efficient nearest-neighbor and range queries in KD-trees.
#[derive(Clone, Debug)]
pub struct Rectangle<T: Float> {
    /// Minimum coordinates for each dimension
    mins: Vec<T>,
    /// Maximum coordinates for each dimension
    maxes: Vec<T>,
}

impl<T: Float> Rectangle<T> {
    /// Create a new hyperrectangle
    ///
    /// # Arguments
    ///
    /// * `mins` - Minimum coordinates for each dimension
    /// * `maxes` - Maximum coordinates for each dimension
    ///
    /// # Returns
    ///
    /// * A new rectangle
    ///
    /// # Panics
    ///
    /// * If mins and maxes have different lengths
    /// * If any min value is greater than the corresponding max value
    pub fn new(mins: Vec<T>, maxes: Vec<T>) -> Self {
        assert_eq!(
            mins.len(),
            maxes.len(),
            "mins and maxes must have the same length"
        );

        for i in 0..mins.len() {
            assert!(
                mins[i] <= maxes[i],
                "min value must be less than or equal to max value"
            );
        }

        Rectangle { mins, maxes }
    }

    /// Get the minimum coordinates of the rectangle
    ///
    /// # Returns
    ///
    /// * A slice containing the minimum coordinate values for each dimension
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::kdtree::Rectangle;
    ///
    /// let rect = Rectangle::new(vec![0.0, 0.0], vec![1.0, 1.0]);
    /// let mins = rect.mins();
    /// assert_eq!(mins, &[0.0, 0.0]);
    /// ```
    pub fn mins(&self) -> &[T] {
        &self.mins
    }

    /// Get the maximum coordinates of the rectangle
    ///
    /// # Returns
    ///
    /// * A slice containing the maximum coordinate values for each dimension
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::kdtree::Rectangle;
    ///
    /// let rect = Rectangle::new(vec![0.0, 0.0], vec![1.0, 1.0]);
    /// let maxes = rect.maxes();
    /// assert_eq!(maxes, &[1.0, 1.0]);
    /// ```
    pub fn maxes(&self) -> &[T] {
        &self.maxes
    }

    /// Split the rectangle along a given dimension at a given value
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension to split on
    /// * `value` - The value to split at
    ///
    /// # Returns
    ///
    /// * A tuple of (left, right) rectangles
    pub fn split(&self, dim: usize, value: T) -> (Self, Self) {
        let mut left_maxes = self.maxes.clone();
        left_maxes[dim] = value;

        let mut right_mins = self.mins.clone();
        right_mins[dim] = value;

        let left = Rectangle::new(self.mins.clone(), left_maxes);
        let right = Rectangle::new(right_mins, self.maxes.clone());

        (left, right)
    }

    /// Check if the rectangle contains a point
    ///
    /// # Arguments
    ///
    /// * `point` - The point to check
    ///
    /// # Returns
    ///
    /// * true if the rectangle contains the point, false otherwise
    pub fn contains(&self, point: &[T]) -> bool {
        assert_eq!(
            point.len(),
            self.mins.len(),
            "point must have the same dimension as the rectangle"
        );

        for (i, &p) in point.iter().enumerate() {
            if p < self.mins[i] || p > self.maxes[i] {
                return false;
            }
        }

        true
    }

    /// Calculate the minimum distance from a point to the rectangle
    ///
    /// # Arguments
    ///
    /// * `point` - The point to calculate distance to
    /// * `metric` - The distance metric to use
    ///
    /// # Returns
    ///
    /// * The minimum distance from the point to any point in the rectangle
    pub fn min_distance<D: Distance<T>>(&self, point: &[T], metric: &D) -> T {
        metric.min_distance_point_rectangle(point, &self.mins, &self.maxes)
    }
}

/// A node in the KD-Tree
#[derive(Debug, Clone)]
struct KDNode<T: Float> {
    /// Index of the point in the original data array
    idx: usize,
    /// The value of the point along the splitting dimension
    value: T,
    /// The dimension used for splitting
    axis: usize,
    /// Left child node (values < median along splitting axis)
    left: Option<usize>,
    /// Right child node (values >= median along splitting axis)
    right: Option<usize>,
}

/// A KD-Tree for efficient nearest neighbor searches
///
/// # Type Parameters
///
/// * `T` - The floating point type for coordinates
/// * `D` - The distance metric type
#[derive(Debug, Clone)]
pub struct KDTree<T: Float + Send + Sync + 'static, D: Distance<T> + 'static> {
    /// The points stored in the KD-Tree
    points: Array2<T>,
    /// The nodes of the KD-Tree
    nodes: Vec<KDNode<T>>,
    /// The dimensionality of the points
    ndim: usize,
    /// The root node index
    root: Option<usize>,
    /// The distance metric
    metric: D,
    /// The leaf size (maximum number of points in a leaf node)
    leafsize: usize,
    /// Minimum bounding rectangle of the entire dataset
    bounds: Rectangle<T>,
}

impl<T: Float + Send + Sync + 'static> KDTree<T, EuclideanDistance<T>> {
    /// Create a new KD-Tree with default Euclidean distance metric
    ///
    /// # Arguments
    ///
    /// * `points` - Array of points, each row is a point
    ///
    /// # Returns
    ///
    /// * A new KD-Tree
    pub fn new(points: &Array2<T>) -> SpatialResult<Self> {
        let metric = EuclideanDistance::new();
        Self::with_metric(points, metric)
    }

    /// Create a new KD-Tree with custom leaf size (using Euclidean distance)
    ///
    /// # Arguments
    ///
    /// * `points` - Array of points, each row is a point
    /// * `leafsize` - The maximum number of points in a leaf node
    ///
    /// # Returns
    ///
    /// * A new KD-Tree
    pub fn with_leaf_size(points: &Array2<T>, leafsize: usize) -> SpatialResult<Self> {
        let metric = EuclideanDistance::new();
        Self::with_options(points, metric, leafsize)
    }
}

impl<T: Float + Send + Sync + 'static, D: Distance<T> + 'static> KDTree<T, D> {
    /// Create a new KD-Tree with custom distance metric
    ///
    /// # Arguments
    ///
    /// * `points` - Array of points, each row is a point
    /// * `metric` - The distance metric to use
    ///
    /// # Returns
    ///
    /// * A new KD-Tree
    pub fn with_metric(points: &Array2<T>, metric: D) -> SpatialResult<Self> {
        Self::with_options(points, metric, 16) // Default leaf size is 16
    }

    /// Create a new KD-Tree with custom distance metric and leaf size
    ///
    /// # Arguments
    ///
    /// * `points` - Array of points, each row is a point
    /// * `metric` - The distance metric to use
    /// * `leafsize` - The maximum number of points in a leaf node
    ///
    /// # Returns
    ///
    /// * A new KD-Tree
    pub fn with_options(points: &Array2<T>, metric: D, leafsize: usize) -> SpatialResult<Self> {
        let n = points.nrows();
        let ndim = points.ncols();

        if n == 0 {
            return Err(SpatialError::ValueError("Empty point set".to_string()));
        }

        if leafsize == 0 {
            return Err(SpatialError::ValueError(
                "Leaf _size must be greater than 0".to_string(),
            ));
        }

        // Calculate the bounds of the dataset
        let mut mins = vec![T::max_value(); ndim];
        let mut maxes = vec![T::min_value(); ndim];

        for i in 0..n {
            for j in 0..ndim {
                let val = points[[i, j]];
                if val < mins[j] {
                    mins[j] = val;
                }
                if val > maxes[j] {
                    maxes[j] = val;
                }
            }
        }

        let bounds = Rectangle::new(mins, maxes);

        let mut tree = KDTree {
            points: points.clone(),
            nodes: Vec::with_capacity(n),
            ndim,
            root: None,
            metric,
            leafsize,
            bounds,
        };

        // Create indices for the points
        let mut indices: Vec<usize> = (0..n).collect();

        // Build the tree recursively
        if n > 0 {
            let root = tree.build_tree(&mut indices, 0, 0, n)?;
            tree.root = Some(root);
        }

        Ok(tree)
    }

    /// Build the KD-Tree recursively
    ///
    /// # Arguments
    ///
    /// * `indices` - Indices of the points to consider
    /// * `depth` - Current depth in the tree
    /// * `start` - Start index in the indices array
    /// * `end` - End index in the indices array
    ///
    /// # Returns
    ///
    /// * Index of the root node of the subtree
    fn build_tree(
        &mut self,
        indices: &mut [usize],
        depth: usize,
        start: usize,
        end: usize,
    ) -> SpatialResult<usize> {
        let n = end - start;

        if n == 0 {
            return Err(SpatialError::ValueError(
                "Empty point set in build_tree".to_string(),
            ));
        }

        // Choose axis based on depth (cycle through axes)
        let axis = depth % self.ndim;

        // If we have only one point, create a leaf node
        let node_idx;
        if n == 1 {
            let idx = indices[start];
            let value = self.points[[idx, axis]];

            node_idx = self.nodes.len();
            self.nodes.push(KDNode {
                idx,
                value,
                axis,
                left: None,
                right: None,
            });

            return Ok(node_idx);
        }

        // Sort indices based on the axis
        indices[start..end].sort_by(|&i, &j| {
            let a = self.points[[i, axis]];
            let b = self.points[[j, axis]];
            a.partial_cmp(&b).unwrap_or(Ordering::Equal)
        });

        // Get the median index
        let mid = start + n / 2;
        let idx = indices[mid];
        let value = self.points[[idx, axis]];

        // Create node
        node_idx = self.nodes.len();
        self.nodes.push(KDNode {
            idx,
            value,
            axis,
            left: None,
            right: None,
        });

        // Build left and right subtrees
        if mid > start {
            let left_idx = self.build_tree(indices, depth + 1, start, mid)?;
            self.nodes[node_idx].left = Some(left_idx);
        }

        if mid + 1 < end {
            let right_idx = self.build_tree(indices, depth + 1, mid + 1, end)?;
            self.nodes[node_idx].right = Some(right_idx);
        }

        Ok(node_idx)
    }

    /// Find the k nearest neighbors to a query point
    ///
    /// # Arguments
    ///
    /// * `point` - Query point
    /// * `k` - Number of nearest neighbors to find
    ///
    /// # Returns
    ///
    /// * (indices, distances) of the k nearest neighbors
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::KDTree;
    /// use scirs2_core::ndarray::array;
    ///
    /// // Create points for the KDTree - use the exact same points from test_kdtree_with_custom_leaf_size
    /// let points = array![[2.0, 3.0], [5.0, 4.0], [9.0, 6.0], [4.0, 7.0], [8.0, 1.0], [7.0, 2.0]];
    /// let kdtree = KDTree::new(&points).unwrap();
    ///
    /// // Find the 2 nearest neighbors to [0.5, 0.5]
    /// let (indices, distances) = kdtree.query(&[0.5, 0.5], 2).unwrap();
    /// assert_eq!(indices.len(), 2);
    /// assert_eq!(distances.len(), 2);
    /// ```
    pub fn query(&self, point: &[T], k: usize) -> SpatialResult<(Vec<usize>, Vec<T>)> {
        if point.len() != self.ndim {
            return Err(SpatialError::DimensionError(format!(
                "Query point dimension ({}) does not match tree dimension ({})",
                point.len(),
                self.ndim
            )));
        }

        if k == 0 {
            return Ok((vec![], vec![]));
        }

        if self.points.nrows() == 0 {
            return Ok((vec![], vec![]));
        }

        // Initialize priority queue for k nearest neighbors
        // We use a max-heap so we can efficiently replace the furthest point when we find a closer one
        let mut neighbors: Vec<(T, usize)> = Vec::with_capacity(k + 1);

        // Keep track of the maximum distance in the heap, for early termination
        let mut max_dist = T::infinity();

        if let Some(root) = self.root {
            // Search recursively
            self.query_recursive(root, point, k, &mut neighbors, &mut max_dist);

            // Sort by distance (ascending), with index as tiebreaker
            neighbors.sort_by(|a, b| {
                match safe_partial_cmp(&a.0, &b.0, "kdtree sort neighbors") {
                    Ok(std::cmp::Ordering::Equal) => a.1.cmp(&b.1), // Use index as tiebreaker
                    Ok(ord) => ord,
                    Err(_) => std::cmp::Ordering::Equal,
                }
            });

            // Trim to k elements if needed
            if neighbors.len() > k {
                neighbors.truncate(k);
            }

            // Convert to sorted lists of indices and distances
            let mut indices = Vec::with_capacity(neighbors.len());
            let mut distances = Vec::with_capacity(neighbors.len());

            for (dist, idx) in neighbors {
                indices.push(idx);
                distances.push(dist);
            }

            Ok((indices, distances))
        } else {
            Err(SpatialError::ValueError("Empty tree".to_string()))
        }
    }

    /// Recursive helper for query
    fn query_recursive(
        &self,
        node_idx: usize,
        point: &[T],
        k: usize,
        neighbors: &mut Vec<(T, usize)>,
        max_dist: &mut T,
    ) {
        let node = &self.nodes[node_idx];
        let idx = node.idx;
        let axis = node.axis;

        // Calculate distance to current point
        let node_point = self.points.row(idx).to_vec();
        let _dist = self.metric.distance(&node_point, point);

        // Update neighbors if needed
        if neighbors.len() < k {
            neighbors.push((_dist, idx));

            // Sort if we just filled to capacity to establish max-heap
            if neighbors.len() == k {
                neighbors.sort_by(|a, b| {
                    match safe_partial_cmp(&b.0, &a.0, "kdtree sort max-heap") {
                        Ok(std::cmp::Ordering::Equal) => b.1.cmp(&a.1), // Use index as tiebreaker
                        Ok(ord) => ord,
                        Err(_) => std::cmp::Ordering::Equal,
                    }
                });
                *max_dist = neighbors[0].0;
            }
        } else if &_dist < max_dist {
            // Replace the worst neighbor with this one
            neighbors[0] = (_dist, idx);

            // Re-sort to maintain max-heap property
            neighbors.sort_by(|a, b| {
                match safe_partial_cmp(&b.0, &a.0, "kdtree re-sort max-heap") {
                    Ok(std::cmp::Ordering::Equal) => b.1.cmp(&a.1), // Use index as tiebreaker
                    Ok(ord) => ord,
                    Err(_) => std::cmp::Ordering::Equal,
                }
            });
            *max_dist = neighbors[0].0;
        }

        // Determine which subtree to search first
        let diff = point[axis] - node.value;
        let (first, second) = if diff < T::zero() {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };

        // Search the near subtree
        if let Some(first_idx) = first {
            self.query_recursive(first_idx, point, k, neighbors, max_dist);
        }

        // Only search the far subtree if it could contain closer points
        let axis_dist = if diff < T::zero() {
            // Point is to the left of the splitting hyperplane
            T::zero() // No need to calculate distance if we're considering the left subtree next
        } else {
            // Point is to the right of the splitting hyperplane
            diff
        };

        if let Some(second_idx) = second {
            // Only search the second subtree if necessary
            if neighbors.len() < k || axis_dist < *max_dist {
                self.query_recursive(second_idx, point, k, neighbors, max_dist);
            }
        }
    }

    /// Find all points within a radius of a query point
    ///
    /// # Arguments
    ///
    /// * `point` - Query point
    /// * `radius` - Search radius
    ///
    /// # Returns
    ///
    /// * (indices, distances) of points within the radius
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::KDTree;
    /// use scirs2_core::ndarray::array;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    /// let kdtree = KDTree::new(&points)?;
    ///
    /// // Find all points within radius 0.7 of [0.5, 0.5]
    /// let (indices, distances) = kdtree.query_radius(&[0.5, 0.5], 0.7)?;
    /// assert_eq!(indices.len(), 4); // All points are within 0.7 units of [0.5, 0.5]
    /// # Ok(())
    /// # }
    /// ```
    pub fn query_radius(&self, point: &[T], radius: T) -> SpatialResult<(Vec<usize>, Vec<T>)> {
        if point.len() != self.ndim {
            return Err(SpatialError::DimensionError(format!(
                "Query point dimension ({}) does not match tree dimension ({})",
                point.len(),
                self.ndim
            )));
        }

        if radius < T::zero() {
            return Err(SpatialError::ValueError(
                "Radius must be non-negative".to_string(),
            ));
        }

        let mut indices = Vec::new();
        let mut distances = Vec::new();

        if let Some(root) = self.root {
            // If the radius is outside the bounds of the entire dataset, just return an empty result
            let bounds_dist = self.bounds.min_distance(point, &self.metric);
            if bounds_dist > radius {
                return Ok((indices, distances));
            }

            // Search recursively
            self.query_radius_recursive(root, point, radius, &mut indices, &mut distances);

            // Sort by distance
            if !indices.is_empty() {
                let mut idx_dist: Vec<(usize, T)> = indices.into_iter().zip(distances).collect();
                idx_dist.sort_by(|a, b| {
                    safe_partial_cmp(&a.1, &b.1, "kdtree sort radius results")
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                indices = idx_dist.iter().map(|(idx_, _)| *idx_).collect();
                distances = idx_dist.iter().map(|(_, dist)| *dist).collect();
            }
        }

        Ok((indices, distances))
    }

    /// Recursive helper for query_radius
    fn query_radius_recursive(
        &self,
        node_idx: usize,
        point: &[T],
        radius: T,
        indices: &mut Vec<usize>,
        distances: &mut Vec<T>,
    ) {
        let node = &self.nodes[node_idx];
        let idx = node.idx;
        let axis = node.axis;

        // Calculate distance to current point
        let node_point = self.points.row(idx).to_vec();
        let dist = self.metric.distance(&node_point, point);

        // If point is within radius, add it to results
        if dist <= radius {
            indices.push(idx);
            distances.push(dist);
        }

        // Determine which subtrees need to be searched
        let diff = point[axis] - node.value;

        // Always search the near subtree
        let (near, far) = if diff < T::zero() {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };

        if let Some(near_idx) = near {
            self.query_radius_recursive(near_idx, point, radius, indices, distances);
        }

        // Only search the far subtree if it could contain points within radius
        if diff.abs() <= radius {
            if let Some(far_idx) = far {
                self.query_radius_recursive(far_idx, point, radius, indices, distances);
            }
        }
    }

    /// Count the number of points within a radius of a query point
    ///
    /// This method is more efficient than query_radius when only the count is needed.
    ///
    /// # Arguments
    ///
    /// * `point` - Query point
    /// * `radius` - Search radius
    ///
    /// # Returns
    ///
    /// * Number of points within the radius
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::KDTree;
    /// use scirs2_core::ndarray::array;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    /// let kdtree = KDTree::new(&points)?;
    ///
    /// // Count points within radius 0.7 of [0.5, 0.5]
    /// let count = kdtree.count_neighbors(&[0.5, 0.5], 0.7)?;
    /// assert_eq!(count, 4); // All points are within 0.7 units of [0.5, 0.5]
    /// # Ok(())
    /// # }
    /// ```
    pub fn count_neighbors(&self, point: &[T], radius: T) -> SpatialResult<usize> {
        if point.len() != self.ndim {
            return Err(SpatialError::DimensionError(format!(
                "Query point dimension ({}) does not match tree dimension ({})",
                point.len(),
                self.ndim
            )));
        }

        if radius < T::zero() {
            return Err(SpatialError::ValueError(
                "Radius must be non-negative".to_string(),
            ));
        }

        let mut count = 0;

        if let Some(root) = self.root {
            // If the radius is outside the bounds of the entire dataset, just return 0
            let bounds_dist = self.bounds.min_distance(point, &self.metric);
            if bounds_dist > radius {
                return Ok(0);
            }

            // Search recursively
            self.count_neighbors_recursive(root, point, radius, &mut count);
        }

        Ok(count)
    }

    /// Recursive helper for count_neighbors
    fn count_neighbors_recursive(
        &self,
        node_idx: usize,
        point: &[T],
        radius: T,
        count: &mut usize,
    ) {
        let node = &self.nodes[node_idx];
        let idx = node.idx;
        let axis = node.axis;

        // Calculate distance to current point
        let node_point = self.points.row(idx).to_vec();
        let dist = self.metric.distance(&node_point, point);

        // If point is within radius, increment count
        if dist <= radius {
            *count += 1;
        }

        // Determine which subtrees need to be searched
        let diff = point[axis] - node.value;

        // Always search the near subtree
        let (near, far) = if diff < T::zero() {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };

        if let Some(near_idx) = near {
            self.count_neighbors_recursive(near_idx, point, radius, count);
        }

        // Only search the far subtree if it could contain points within radius
        if diff.abs() <= radius {
            if let Some(far_idx) = far {
                self.count_neighbors_recursive(far_idx, point, radius, count);
            }
        }
    }

    /// Get the shape of the KD-Tree's point set
    ///
    /// # Returns
    ///
    /// * A tuple of (n_points, n_dimensions)
    pub fn shape(&self) -> (usize, usize) {
        (self.points.nrows(), self.ndim)
    }

    /// Get the number of points in the KD-Tree
    ///
    /// # Returns
    ///
    /// * The number of points
    pub fn npoints(&self) -> usize {
        self.points.nrows()
    }

    /// Get the dimensionality of the points in the KD-Tree
    ///
    /// # Returns
    ///
    /// * The dimensionality of the points
    pub fn ndim(&self) -> usize {
        self.ndim
    }

    /// Get the leaf size of the KD-Tree
    ///
    /// # Returns
    ///
    /// * The leaf size
    pub fn leafsize(&self) -> usize {
        self.leafsize
    }

    /// Get the bounds of the KD-Tree
    ///
    /// # Returns
    ///
    /// * The bounding rectangle of the entire dataset
    pub fn bounds(&self) -> &Rectangle<T> {
        &self.bounds
    }
}

#[cfg(test)]
mod tests {
    use super::{KDTree, Rectangle};
    use crate::distance::{
        ChebyshevDistance, Distance, EuclideanDistance, ManhattanDistance, MinkowskiDistance,
    };
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::arr2;

    #[test]
    fn test_rectangle() {
        let mins = vec![0.0, 0.0];
        let maxes = vec![1.0, 1.0];
        let rect = Rectangle::new(mins, maxes);

        // Test contains
        assert!(rect.contains(&[0.5, 0.5]));
        assert!(rect.contains(&[0.0, 0.0]));
        assert!(rect.contains(&[1.0, 1.0]));
        assert!(!rect.contains(&[1.5, 0.5]));
        assert!(!rect.contains(&[0.5, 1.5]));

        // Test split
        let (left, right) = rect.split(0, 0.5);
        assert!(left.contains(&[0.25, 0.5]));
        assert!(!left.contains(&[0.75, 0.5]));
        assert!(!right.contains(&[0.25, 0.5]));
        assert!(right.contains(&[0.75, 0.5]));

        // Test min_distance
        let metric = EuclideanDistance::<f64>::new();
        assert_relative_eq!(rect.min_distance(&[0.5, 0.5], &metric), 0.0, epsilon = 1e-6);
        assert_relative_eq!(rect.min_distance(&[2.0, 0.5], &metric), 1.0, epsilon = 1e-6);
        assert_relative_eq!(
            rect.min_distance(&[2.0, 2.0], &metric),
            std::f64::consts::SQRT_2,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_kdtree_build() {
        let points = arr2(&[
            [2.0, 3.0],
            [5.0, 4.0],
            [9.0, 6.0],
            [4.0, 7.0],
            [8.0, 1.0],
            [7.0, 2.0],
        ]);

        let kdtree = KDTree::new(&points).unwrap();

        // Check that the tree has the correct number of nodes
        assert_eq!(kdtree.nodes.len(), points.nrows());

        // Check tree properties
        assert_eq!(kdtree.shape(), (6, 2));
        assert_eq!(kdtree.npoints(), 6);
        assert_eq!(kdtree.ndim(), 2);
        assert_eq!(kdtree.leafsize(), 16);

        // Check bounds
        assert_eq!(kdtree.bounds().mins(), &[2.0, 1.0]);
        assert_eq!(kdtree.bounds().maxes(), &[9.0, 7.0]);
    }

    #[test]
    fn test_kdtree_query() {
        let points = arr2(&[
            [2.0, 3.0],
            [5.0, 4.0],
            [9.0, 6.0],
            [4.0, 7.0],
            [8.0, 1.0],
            [7.0, 2.0],
        ]);

        let kdtree = KDTree::new(&points).unwrap();

        // Query for nearest neighbor to [3.0, 5.0]
        let (indices, distances) = kdtree.query(&[3.0, 5.0], 1).unwrap();
        assert_eq!(indices.len(), 1);
        assert_eq!(distances.len(), 1);

        // Calculate actual distances
        let query = [3.0, 5.0];
        let mut expected_dists = vec![];
        for i in 0..points.nrows() {
            let p = points.row(i).to_vec();
            let metric = EuclideanDistance::<f64>::new();
            expected_dists.push((i, metric.distance(&p, &query)));
        }
        expected_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Verify we got one of the actual nearest neighbors (there might be ties)
        // Check that the distance matches the expected minimum distance
        assert_relative_eq!(distances[0], expected_dists[0].1, epsilon = 1e-6);

        // Verify the returned index is one of the points with minimum distance
        let min_dist = expected_dists[0].1;
        let valid_indices: Vec<usize> = expected_dists
            .iter()
            .filter(|(_, d)| (d - min_dist).abs() < 1e-6)
            .map(|(i, _)| *i)
            .collect();
        assert!(
            valid_indices.contains(&indices[0]),
            "Expected one of {:?} but got {}",
            valid_indices,
            indices[0]
        );
    }

    #[test]
    fn test_kdtree_query_k() {
        let points = arr2(&[
            [2.0, 3.0],
            [5.0, 4.0],
            [9.0, 6.0],
            [4.0, 7.0],
            [8.0, 1.0],
            [7.0, 2.0],
        ]);

        let kdtree = KDTree::new(&points).unwrap();

        // Query for 3 nearest neighbors to [3.0, 5.0]
        let (indices, distances) = kdtree.query(&[3.0, 5.0], 3).unwrap();
        assert_eq!(indices.len(), 3);
        assert_eq!(distances.len(), 3);

        // Calculate actual distances
        let query = [3.0, 5.0];
        let mut expected_dists = vec![];
        for i in 0..points.nrows() {
            let p = points.row(i).to_vec();
            let metric = EuclideanDistance::<f64>::new();
            expected_dists.push((i, metric.distance(&p, &query)));
        }
        expected_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Verify we got the 3 actual nearest neighbors (for now, just check distances)
        let expected_indices: Vec<usize> = expected_dists.iter().take(3).map(|&(i, _)| i).collect();
        let expected_distances: Vec<f64> = expected_dists.iter().take(3).map(|&(_, d)| d).collect();

        // Check each returned index is in the expected set
        for i in &indices {
            assert!(expected_indices.contains(i));
        }

        // Check that distances are sorted
        assert!(distances[0] <= distances[1]);
        assert!(distances[1] <= distances[2]);

        // Check distance values match expected
        for i in 0..3 {
            assert_relative_eq!(distances[i], expected_distances[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_kdtree_query_radius() {
        let points = arr2(&[
            [2.0, 3.0],
            [5.0, 4.0],
            [9.0, 6.0],
            [4.0, 7.0],
            [8.0, 1.0],
            [7.0, 2.0],
        ]);

        let kdtree = KDTree::new(&points).unwrap();

        // Query for points within radius 3.0 of [3.0, 5.0]
        let (indices, distances) = kdtree.query_radius(&[3.0, 5.0], 3.0).unwrap();

        // Calculate expected results
        let query = [3.0, 5.0];
        let radius = 3.0;
        let mut expected_results = vec![];
        for i in 0..points.nrows() {
            let p = points.row(i).to_vec();
            let metric = EuclideanDistance::<f64>::new();
            let dist = metric.distance(&p, &query);
            if dist <= radius {
                expected_results.push((i, dist));
            }
        }
        expected_results.sort_by(|a, b| {
            match a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal) {
                std::cmp::Ordering::Equal => a.0.cmp(&b.0), // Use index as tiebreaker
                ord => ord,
            }
        });

        // Check that we got the expected number of points
        assert_eq!(indices.len(), expected_results.len());

        // Check that all returned points are within radius
        for i in 0..indices.len() {
            assert!(distances[i] <= radius + 1e-6);
        }

        // Check that the indices/distances pairs match expected results
        // Note: order might differ for equal distances
        let mut idx_dist_pairs: Vec<(usize, f64)> = indices
            .iter()
            .zip(distances.iter())
            .map(|(&i, &d)| (i, d))
            .collect();
        idx_dist_pairs.sort_by(|a, b| {
            match a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal) {
                std::cmp::Ordering::Equal => a.0.cmp(&b.0),
                ord => ord,
            }
        });

        for (actual, expected) in idx_dist_pairs.iter().zip(expected_results.iter()) {
            assert_eq!(actual.0, expected.0);
            assert_relative_eq!(actual.1, expected.1, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_kdtree_count_neighbors() {
        let points = arr2(&[
            [2.0, 3.0],
            [5.0, 4.0],
            [9.0, 6.0],
            [4.0, 7.0],
            [8.0, 1.0],
            [7.0, 2.0],
        ]);

        let kdtree = KDTree::new(&points).unwrap();

        // Count points within radius 3.0 of [3.0, 5.0]
        let count = kdtree.count_neighbors(&[3.0, 5.0], 3.0).unwrap();

        // Calculate actual count
        let query = [3.0, 5.0];
        let mut expected_count = 0;
        for i in 0..points.nrows() {
            let p = points.row(i).to_vec();
            let metric = EuclideanDistance::<f64>::new();
            let dist = metric.distance(&p, &query);
            if dist <= 3.0 {
                expected_count += 1;
            }
        }

        assert_eq!(count, expected_count);
    }

    #[test]
    fn test_kdtree_with_manhattan_distance() {
        let points = arr2(&[
            [2.0, 3.0],
            [5.0, 4.0],
            [9.0, 6.0],
            [4.0, 7.0],
            [8.0, 1.0],
            [7.0, 2.0],
        ]);

        let metric = ManhattanDistance::new();
        let kdtree = KDTree::with_metric(&points, metric).unwrap();

        // Query for nearest neighbor to [3.0, 5.0] using Manhattan distance
        let (indices, distances) = kdtree.query(&[3.0, 5.0], 1).unwrap();

        // Calculate actual distances using Manhattan distance
        let query = [3.0, 5.0];
        let mut expected_dists = vec![];
        for i in 0..points.nrows() {
            let p = points.row(i).to_vec();
            let m = ManhattanDistance::<f64>::new();
            expected_dists.push((i, m.distance(&p, &query)));
        }
        expected_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Check that the distance matches the expected minimum distance
        assert_relative_eq!(distances[0], expected_dists[0].1, epsilon = 1e-6);

        // Verify the returned index is one of the points with minimum distance
        let min_dist = expected_dists[0].1;
        let valid_indices: Vec<usize> = expected_dists
            .iter()
            .filter(|(_, d)| (d - min_dist).abs() < 1e-6)
            .map(|(i, _)| *i)
            .collect();
        assert!(
            valid_indices.contains(&indices[0]),
            "Expected one of {:?} but got {}",
            valid_indices,
            indices[0]
        );
    }

    #[test]
    fn test_kdtree_with_chebyshev_distance() {
        let points = arr2(&[
            [2.0, 3.0],
            [5.0, 4.0],
            [9.0, 6.0],
            [4.0, 7.0],
            [8.0, 1.0],
            [7.0, 2.0],
        ]);

        let metric = ChebyshevDistance::new();
        let kdtree = KDTree::with_metric(&points, metric).unwrap();

        // Query for nearest neighbor to [3.0, 5.0] using Chebyshev distance
        let (indices, distances) = kdtree.query(&[3.0, 5.0], 1).unwrap();

        // Calculate actual distances using Chebyshev distance
        let query = [3.0, 5.0];
        let mut expected_dists = vec![];
        for i in 0..points.nrows() {
            let p = points.row(i).to_vec();
            let m = ChebyshevDistance::<f64>::new();
            expected_dists.push((i, m.distance(&p, &query)));
        }
        expected_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Check that the distance matches the expected minimum distance
        assert_relative_eq!(distances[0], expected_dists[0].1, epsilon = 1e-6);

        // Verify the returned index is one of the points with minimum distance
        let min_dist = expected_dists[0].1;
        let valid_indices: Vec<usize> = expected_dists
            .iter()
            .filter(|(_, d)| (d - min_dist).abs() < 1e-6)
            .map(|(i, _)| *i)
            .collect();
        assert!(
            valid_indices.contains(&indices[0]),
            "Expected one of {:?} but got {}",
            valid_indices,
            indices[0]
        );
    }

    #[test]
    fn test_kdtree_with_minkowski_distance() {
        let points = arr2(&[
            [2.0, 3.0],
            [5.0, 4.0],
            [9.0, 6.0],
            [4.0, 7.0],
            [8.0, 1.0],
            [7.0, 2.0],
        ]);

        let metric = MinkowskiDistance::new(3.0);
        let kdtree = KDTree::with_metric(&points, metric).unwrap();

        // Query for nearest neighbor to [3.0, 5.0] using Minkowski distance (p=3)
        let (indices, distances) = kdtree.query(&[3.0, 5.0], 1).unwrap();

        // Calculate actual distances using Minkowski distance
        let query = [3.0, 5.0];
        let mut expected_dists = vec![];
        for i in 0..points.nrows() {
            let p = points.row(i).to_vec();
            let m = MinkowskiDistance::<f64>::new(3.0);
            expected_dists.push((i, m.distance(&p, &query)));
        }
        expected_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Check that the distance matches the expected minimum distance
        assert_relative_eq!(distances[0], expected_dists[0].1, epsilon = 1e-6);

        // Verify the returned index is one of the points with minimum distance
        let min_dist = expected_dists[0].1;
        let valid_indices: Vec<usize> = expected_dists
            .iter()
            .filter(|(_, d)| (d - min_dist).abs() < 1e-6)
            .map(|(i, _)| *i)
            .collect();
        assert!(
            valid_indices.contains(&indices[0]),
            "Expected one of {:?} but got {}",
            valid_indices,
            indices[0]
        );
    }

    #[test]
    fn test_kdtree_with_custom_leaf_size() {
        let points = arr2(&[
            [2.0, 3.0],
            [5.0, 4.0],
            [9.0, 6.0],
            [4.0, 7.0],
            [8.0, 1.0],
            [7.0, 2.0],
        ]);

        // Use a very small leaf size to test that it works
        let leafsize = 1;
        let kdtree = KDTree::with_leaf_size(&points, leafsize).unwrap();

        assert_eq!(kdtree.leafsize(), 1);

        // Query for nearest neighbor to [3.0, 5.0]
        let (indices, distances) = kdtree.query(&[3.0, 5.0], 1).unwrap();

        // Calculate actual distances
        let query = [3.0, 5.0];
        let mut expected_dists = vec![];
        for i in 0..points.nrows() {
            let p = points.row(i).to_vec();
            let metric = EuclideanDistance::<f64>::new();
            expected_dists.push((i, metric.distance(&p, &query)));
        }
        expected_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Verify we got one of the actual nearest neighbors (there might be ties)
        // Check that the distance matches the expected minimum distance
        assert_relative_eq!(distances[0], expected_dists[0].1, epsilon = 1e-6);

        // Verify the returned index is one of the points with minimum distance
        let min_dist = expected_dists[0].1;
        let valid_indices: Vec<usize> = expected_dists
            .iter()
            .filter(|(_, d)| (d - min_dist).abs() < 1e-6)
            .map(|(i, _)| *i)
            .collect();
        assert!(
            valid_indices.contains(&indices[0]),
            "Expected one of {:?} but got {}",
            valid_indices,
            indices[0]
        );
    }
}
