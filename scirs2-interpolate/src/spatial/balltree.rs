//! Ball Tree implementation for efficient nearest neighbor search
//!
//! A Ball Tree is a space-partitioning data structure for organizing points in a
//! k-dimensional space. It divides points into nested hyperspheres, which makes it
//! particularly effective for high-dimensional data.
//!
//! Advantages of Ball Trees over KD-Trees:
//! - Better performance in high dimensions
//! - More efficient for datasets with varying density
//! - Handles elongated clusters well
//!
//! This implementation provides:
//! - Building balanced Ball Trees from point data
//! - Efficient exact nearest neighbor queries
//! - k-nearest neighbor searches
//! - Range queries for all points within a specified radius

use ndarray::Array2;
use num_traits::{Float, FromPrimitive};
use ordered_float::OrderedFloat;
use std::cmp::Ordering;
use std::fmt::Debug;
use std::marker::PhantomData;

use crate::error::{InterpolateError, InterpolateResult};

/// A node in the Ball Tree
#[derive(Debug, Clone)]
struct BallNode<F: Float> {
    /// Indices of the points in this node
    indices: Vec<usize>,

    /// Center of the ball
    center: Vec<F>,

    /// Radius of the ball
    radius: F,

    /// Left child node index
    left: Option<usize>,

    /// Right child node index
    right: Option<usize>,
}

/// Ball Tree for efficient nearest neighbor searches in high dimensions
///
/// # Examples
///
/// ```rust
/// use ndarray::Array2;
/// use scirs2_interpolate::spatial::balltree::BallTree;
///
/// // Create sample 3D points
/// let points = Array2::from_shape_vec((5, 3), vec![
///     0.0, 0.0, 0.0,
///     1.0, 0.0, 0.0,
///     0.0, 1.0, 0.0,
///     0.0, 0.0, 1.0,
///     0.5, 0.5, 0.5,
/// ]).unwrap();
///
/// // Build Ball Tree
/// let ball_tree = BallTree::new(points).unwrap();
///
/// // Find the nearest neighbor to point (0.6, 0.6, 0.6)
/// let query = vec![0.6, 0.6, 0.6];
/// let (idx, distance) = ball_tree.nearest_neighbor(&query).unwrap();
///
/// // idx should be 4 (the point at (0.5, 0.5, 0.5))
/// assert_eq!(idx, 4);
/// ```
#[derive(Debug, Clone)]
pub struct BallTree<F>
where
    F: Float + FromPrimitive + Debug + std::cmp::PartialOrd,
{
    /// The original points used to build the tree
    points: Array2<F>,

    /// Nodes of the Ball Tree
    nodes: Vec<BallNode<F>>,

    /// Root node index
    root: Option<usize>,

    /// The dimension of the space
    dim: usize,

    /// Leaf size (max points in a leaf node)
    leaf_size: usize,

    /// Marker for generic type parameter
    _phantom: PhantomData<F>,
}

impl<F> BallTree<F>
where
    F: Float + FromPrimitive + Debug + std::cmp::PartialOrd,
{
    /// Create a new Ball Tree from points
    ///
    /// # Arguments
    ///
    /// * `points` - Point coordinates with shape (n_points, n_dims)
    ///
    /// # Returns
    ///
    /// A new Ball Tree for efficient nearest neighbor searches
    pub fn new(points: Array2<F>) -> InterpolateResult<Self> {
        Self::with_leaf_size(points, 10)
    }

    /// Create a new Ball Tree with a specified leaf size
    ///
    /// # Arguments
    ///
    /// * `points` - Point coordinates with shape (n_points, n_dims)
    /// * `leaf_size` - Maximum number of points in a leaf node
    ///
    /// # Returns
    ///
    /// A new Ball Tree for efficient nearest neighbor searches
    pub fn with_leaf_size(points: Array2<F>, leaf_size: usize) -> InterpolateResult<Self> {
        if points.is_empty() {
            return Err(InterpolateError::InvalidValue(
                "Points array cannot be empty".to_string(),
            ));
        }

        let n_points = points.shape()[0];
        let dim = points.shape()[1];

        // For very small datasets, just use a simple linear search
        if n_points <= leaf_size {
            let indices: Vec<usize> = (0..n_points).collect();
            let center = compute_centroid(&points, &indices);
            let radius = compute_radius(&points, &indices, &center);

            let mut tree = Self {
                points,
                nodes: Vec::new(),
                root: None,
                dim,
                leaf_size,
                _phantom: PhantomData,
            };

            if n_points > 0 {
                // Create a single root node
                tree.nodes.push(BallNode {
                    indices,
                    center,
                    radius,
                    left: None,
                    right: None,
                });
                tree.root = Some(0);
            }

            return Ok(tree);
        }

        // Pre-allocate nodes (approximately 2*n_points/leaf_size)
        let est_nodes = (2 * n_points / leaf_size).max(16);

        let mut tree = Self {
            points,
            nodes: Vec::with_capacity(est_nodes),
            root: None,
            dim,
            leaf_size,
            _phantom: PhantomData,
        };

        // Build the tree
        let indices: Vec<usize> = (0..n_points).collect();
        tree.root = Some(tree.build_subtree(&indices));

        Ok(tree)
    }

    /// Build a subtree recursively
    fn build_subtree(&mut self, indices: &[usize]) -> usize {
        let n_points = indices.len();

        // Compute center and radius of this ball
        let center = compute_centroid(&self.points, indices);
        let radius = compute_radius(&self.points, indices, &center);

        // If few enough points, create a leaf node
        if n_points <= self.leaf_size {
            let node_idx = self.nodes.len();
            self.nodes.push(BallNode {
                indices: indices.to_vec(),
                center,
                radius,
                left: None,
                right: None,
            });
            return node_idx;
        }

        // Find the dimension with the largest spread
        let (split_dim, _) = find_max_spread_dimension(&self.points, indices);

        // Find the two points farthest apart along this dimension to use as seeds
        let (seed1, seed2) = find_distant_points(&self.points, indices, split_dim);

        // Partition points based on which seed they're closer to
        let (left_indices, right_indices) = partition_by_seeds(&self.points, indices, seed1, seed2);

        // Create node for this ball
        let node_idx = self.nodes.len();
        self.nodes.push(BallNode {
            indices: indices.to_vec(),
            center,
            radius,
            left: None,
            right: None,
        });

        // Recursively build left and right subtrees
        let left_idx = self.build_subtree(&left_indices);
        let right_idx = self.build_subtree(&right_indices);

        // Update node with child information
        self.nodes[node_idx].left = Some(left_idx);
        self.nodes[node_idx].right = Some(right_idx);

        node_idx
    }

    /// Find the nearest neighbor to a query point
    ///
    /// # Arguments
    ///
    /// * `query` - Query point coordinates
    ///
    /// # Returns
    ///
    /// Tuple containing (point_index, distance) of the nearest neighbor
    pub fn nearest_neighbor(&self, query: &[F]) -> InterpolateResult<(usize, F)> {
        // Check query dimension
        if query.len() != self.dim {
            return Err(InterpolateError::DimensionMismatch(format!(
                "Query dimension {} doesn't match Ball Tree dimension {}",
                query.len(),
                self.dim
            )));
        }

        // Handle empty tree
        if self.root.is_none() {
            return Err(InterpolateError::InvalidState(
                "Ball Tree is empty".to_string(),
            ));
        }

        // Very small trees (just use linear search)
        if self.points.shape()[0] <= self.leaf_size {
            return self.linear_nearest_neighbor(query);
        }

        // Initialize nearest neighbor search
        let mut best_dist = F::infinity();
        let mut best_idx = 0;

        // Start recursive search
        self.search_nearest(self.root.unwrap(), query, &mut best_dist, &mut best_idx);

        Ok((best_idx, best_dist))
    }

    /// Find k nearest neighbors to a query point
    ///
    /// # Arguments
    ///
    /// * `query` - Query point coordinates
    /// * `k` - Number of nearest neighbors to find
    ///
    /// # Returns
    ///
    /// Vector of (point_index, distance) tuples, sorted by distance
    pub fn k_nearest_neighbors(&self, query: &[F], k: usize) -> InterpolateResult<Vec<(usize, F)>> {
        // Check query dimension
        if query.len() != self.dim {
            return Err(InterpolateError::DimensionMismatch(format!(
                "Query dimension {} doesn't match Ball Tree dimension {}",
                query.len(),
                self.dim
            )));
        }

        // Handle empty tree
        if self.root.is_none() {
            return Err(InterpolateError::InvalidState(
                "Ball Tree is empty".to_string(),
            ));
        }

        // Limit k to the number of points
        let k = k.min(self.points.shape()[0]);

        if k == 0 {
            return Ok(Vec::new());
        }

        // Very small trees (just use linear search)
        if self.points.shape()[0] <= self.leaf_size {
            return self.linear_k_nearest_neighbors(query, k);
        }

        // Use a BinaryHeap as a priority queue to keep track of k nearest points
        use std::collections::BinaryHeap;

        let mut heap = BinaryHeap::with_capacity(k + 1);

        // Start recursive search
        self.search_k_nearest(self.root.unwrap(), query, k, &mut heap);

        // Convert heap to sorted vector
        let mut results: Vec<(usize, F)> = heap
            .into_iter()
            .map(|(dist, idx)| (idx, dist.into_inner()))
            .collect();

        // Sort by distance (since heap gives reverse order)
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        Ok(results)
    }

    /// Find all points within a specified radius of a query point
    ///
    /// # Arguments
    ///
    /// * `query` - Query point coordinates
    /// * `radius` - Search radius
    ///
    /// # Returns
    ///
    /// Vector of (point_index, distance) tuples for all points within radius
    pub fn points_within_radius(
        &self,
        query: &[F],
        radius: F,
    ) -> InterpolateResult<Vec<(usize, F)>> {
        // Check query dimension
        if query.len() != self.dim {
            return Err(InterpolateError::DimensionMismatch(format!(
                "Query dimension {} doesn't match Ball Tree dimension {}",
                query.len(),
                self.dim
            )));
        }

        // Handle empty tree
        if self.root.is_none() {
            return Err(InterpolateError::InvalidState(
                "Ball Tree is empty".to_string(),
            ));
        }

        if radius <= F::zero() {
            return Err(InterpolateError::InvalidValue(
                "Radius must be positive".to_string(),
            ));
        }

        // Very small trees (just use linear search)
        if self.points.shape()[0] <= self.leaf_size {
            return self.linear_points_within_radius(query, radius);
        }

        // Store results
        let mut results = Vec::new();

        // Start recursive search
        self.search_radius(self.root.unwrap(), query, radius, &mut results);

        // Sort by distance
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        Ok(results)
    }

    /// Recursively search for the nearest neighbor
    fn search_nearest(
        &self,
        node_idx: usize,
        query: &[F],
        best_dist: &mut F,
        best_idx: &mut usize,
    ) {
        let node = &self.nodes[node_idx];

        // Calculate distance from query to ball center
        let center_dist = euclidean_distance(query, &node.center);

        // If this ball is too far away to contain a better point, skip it
        if center_dist > node.radius + *best_dist {
            return;
        }

        // If this is a leaf node, check all points
        if node.left.is_none() && node.right.is_none() {
            for &idx in &node.indices {
                let point = self.points.row(idx);
                let dist = euclidean_distance(query, &point.to_vec());

                if dist < *best_dist {
                    *best_dist = dist;
                    *best_idx = idx;
                }
            }
            return;
        }

        // Process children
        // Choose the closer ball first to potentially reduce the best_dist sooner
        let left_idx = node.left.unwrap();
        let right_idx = node.right.unwrap();

        let left_node = &self.nodes[left_idx];
        let right_node = &self.nodes[right_idx];

        let left_dist = euclidean_distance(query, &left_node.center);
        let right_dist = euclidean_distance(query, &right_node.center);

        if left_dist < right_dist {
            // Search left child first
            self.search_nearest(left_idx, query, best_dist, best_idx);
            self.search_nearest(right_idx, query, best_dist, best_idx);
        } else {
            // Search right child first
            self.search_nearest(right_idx, query, best_dist, best_idx);
            self.search_nearest(left_idx, query, best_dist, best_idx);
        }
    }

    /// Recursively search for the k nearest neighbors
    #[allow(clippy::type_complexity)]
    fn search_k_nearest(
        &self,
        node_idx: usize,
        query: &[F],
        k: usize,
        heap: &mut std::collections::BinaryHeap<(OrderedFloat<F>, usize)>,
    ) {
        let node = &self.nodes[node_idx];

        // Calculate distance from query to ball center
        let center_dist = euclidean_distance(query, &node.center);

        // Get current kth distance (the farthest point in our current result set)
        let kth_dist = if heap.len() < k {
            F::infinity()
        } else {
            // Peek at the top of the max-heap to get the farthest point
            match heap.peek() {
                Some(&(dist, _)) => dist.into_inner(),
                None => F::infinity(),
            }
        };

        // If this ball is too far away to contain a better point, skip it
        if center_dist > node.radius + kth_dist {
            return;
        }

        // If this is a leaf node, check all points
        if node.left.is_none() && node.right.is_none() {
            for &idx in &node.indices {
                let point = self.points.row(idx);
                let dist = euclidean_distance(query, &point.to_vec());

                // Add to heap
                heap.push((OrderedFloat(dist), idx));

                // If heap is too large, remove the farthest point
                if heap.len() > k {
                    heap.pop();
                }
            }
            return;
        }

        // Process children
        // Choose the closer ball first to potentially reduce the kth_dist sooner
        let left_idx = node.left.unwrap();
        let right_idx = node.right.unwrap();

        let left_node = &self.nodes[left_idx];
        let right_node = &self.nodes[right_idx];

        let left_dist = euclidean_distance(query, &left_node.center);
        let right_dist = euclidean_distance(query, &right_node.center);

        if left_dist < right_dist {
            // Search left child first
            self.search_k_nearest(left_idx, query, k, heap);
            self.search_k_nearest(right_idx, query, k, heap);
        } else {
            // Search right child first
            self.search_k_nearest(right_idx, query, k, heap);
            self.search_k_nearest(left_idx, query, k, heap);
        }
    }

    /// Recursively search for all points within a radius
    fn search_radius(
        &self,
        node_idx: usize,
        query: &[F],
        radius: F,
        results: &mut Vec<(usize, F)>,
    ) {
        let node = &self.nodes[node_idx];

        // Calculate distance from query to ball center
        let center_dist = euclidean_distance(query, &node.center);

        // If this ball is too far away to contain any points within radius, skip it
        if center_dist > node.radius + radius {
            return;
        }

        // If this is a leaf node, check all points
        if node.left.is_none() && node.right.is_none() {
            for &idx in &node.indices {
                let point = self.points.row(idx);
                let dist = euclidean_distance(query, &point.to_vec());

                if dist <= radius {
                    results.push((idx, dist));
                }
            }
            return;
        }

        // Process children
        if let Some(left_idx) = node.left {
            self.search_radius(left_idx, query, radius, results);
        }

        if let Some(right_idx) = node.right {
            self.search_radius(right_idx, query, radius, results);
        }
    }

    /// Linear search for the nearest neighbor (for small datasets or leaf nodes)
    fn linear_nearest_neighbor(&self, query: &[F]) -> InterpolateResult<(usize, F)> {
        let n_points = self.points.shape()[0];

        let mut min_dist = F::infinity();
        let mut min_idx = 0;

        for i in 0..n_points {
            let point = self.points.row(i);
            let dist = euclidean_distance(query, &point.to_vec());

            if dist < min_dist {
                min_dist = dist;
                min_idx = i;
            }
        }

        Ok((min_idx, min_dist))
    }

    /// Linear search for k nearest neighbors (for small datasets or leaf nodes)
    fn linear_k_nearest_neighbors(
        &self,
        query: &[F],
        k: usize,
    ) -> InterpolateResult<Vec<(usize, F)>> {
        let n_points = self.points.shape()[0];
        let k = k.min(n_points); // Limit k to the number of points

        // Calculate all distances
        let mut distances: Vec<(usize, F)> = (0..n_points)
            .map(|i| {
                let point = self.points.row(i);
                let dist = euclidean_distance(query, &point.to_vec());
                (i, dist)
            })
            .collect();

        // Sort by distance
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        // Return k nearest
        distances.truncate(k);
        Ok(distances)
    }

    /// Linear search for points within radius (for small datasets or leaf nodes)
    fn linear_points_within_radius(
        &self,
        query: &[F],
        radius: F,
    ) -> InterpolateResult<Vec<(usize, F)>> {
        let n_points = self.points.shape()[0];

        // Calculate all distances and filter by radius
        let mut results: Vec<(usize, F)> = (0..n_points)
            .filter_map(|i| {
                let point = self.points.row(i);
                let dist = euclidean_distance(query, &point.to_vec());
                if dist <= radius {
                    Some((i, dist))
                } else {
                    None
                }
            })
            .collect();

        // Sort by distance
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        Ok(results)
    }

    /// Get the number of points in the Ball Tree
    pub fn len(&self) -> usize {
        self.points.shape()[0]
    }

    /// Check if the Ball Tree is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the dimension of points in the Ball Tree
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get a reference to the points in the Ball Tree
    pub fn points(&self) -> &Array2<F> {
        &self.points
    }

    /// Find all points within a specified radius (alias for points_within_radius)
    ///
    /// # Arguments
    ///
    /// * `query` - Query point coordinates
    /// * `radius` - Search radius
    ///
    /// # Returns
    ///
    /// Vector of (point_index, distance) tuples for all points within radius
    pub fn radius_neighbors(&self, query: &[F], radius: F) -> InterpolateResult<Vec<(usize, F)>> {
        self.points_within_radius(query, radius)
    }

    /// Find all points within a specified radius using an array view
    ///
    /// # Arguments
    ///
    /// * `query` - Query point coordinates as an array view
    /// * `radius` - Search radius
    ///
    /// # Returns
    ///
    /// Vector of (point_index, distance) tuples for all points within radius
    pub fn radius_neighbors_view(
        &self,
        query: &ndarray::ArrayView1<F>,
        radius: F,
    ) -> InterpolateResult<Vec<(usize, F)>> {
        let query_slice = query.as_slice().ok_or_else(|| {
            InterpolateError::InvalidValue("Query must be contiguous".to_string())
        })?;
        self.points_within_radius(query_slice, radius)
    }

    /// Enhanced k-nearest neighbor search with distance bounds optimization
    ///
    /// This method provides improved performance for k-NN queries by using
    /// tighter distance bounds and more efficient ball pruning strategies.
    ///
    /// # Arguments
    ///
    /// * `query` - Query point coordinates
    /// * `k` - Number of nearest neighbors to find
    /// * `max_distance` - Optional maximum search distance for early termination
    ///
    /// # Returns
    ///
    /// Vector of (point_index, distance) tuples, sorted by distance
    pub fn k_nearest_neighbors_optimized(
        &self,
        query: &[F],
        k: usize,
        max_distance: Option<F>,
    ) -> InterpolateResult<Vec<(usize, F)>> {
        // Check query dimension
        if query.len() != self.dim {
            return Err(InterpolateError::DimensionMismatch(format!(
                "Query dimension {} doesn't match Ball Tree dimension {}",
                query.len(),
                self.dim
            )));
        }

        // Handle empty tree
        if self.root.is_none() {
            return Err(InterpolateError::InvalidState(
                "Ball Tree is empty".to_string(),
            ));
        }

        // Limit k to the number of points
        let k = k.min(self.points.shape()[0]);

        if k == 0 {
            return Ok(Vec::new());
        }

        // Very small trees (just use linear search)
        if self.points.shape()[0] <= self.leaf_size {
            return self.linear_k_nearest_neighbors_optimized(query, k, max_distance);
        }

        use std::collections::BinaryHeap;

        let mut heap = BinaryHeap::with_capacity(k + 1);
        let mut search_radius = max_distance.unwrap_or(F::infinity());

        // Start recursive search with adaptive bounds
        self.search_k_nearest_optimized(
            self.root.unwrap(),
            query,
            k,
            &mut heap,
            &mut search_radius,
        );

        // Convert heap to sorted vector
        let mut results: Vec<(usize, F)> = heap
            .into_iter()
            .map(|(dist, idx)| (idx, dist.into_inner()))
            .collect();

        // Sort by distance (since heap gives reverse order)
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        Ok(results)
    }

    /// Optimized linear k-nearest neighbors search with early termination
    fn linear_k_nearest_neighbors_optimized(
        &self,
        query: &[F],
        k: usize,
        max_distance: Option<F>,
    ) -> InterpolateResult<Vec<(usize, F)>> {
        let n_points = self.points.shape()[0];
        let k = k.min(n_points);
        let max_dist = max_distance.unwrap_or(F::infinity());

        let mut distances: Vec<(usize, F)> = Vec::with_capacity(n_points);

        for i in 0..n_points {
            let point = self.points.row(i);
            let dist = euclidean_distance(query, &point.to_vec());

            // Early termination if distance exceeds maximum
            if dist <= max_dist {
                distances.push((i, dist));
            }
        }

        // Sort by distance
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        // Return k nearest within max distance
        distances.truncate(k);
        Ok(distances)
    }

    /// Optimized recursive k-nearest search with enhanced ball pruning
    #[allow(clippy::type_complexity)]
    fn search_k_nearest_optimized(
        &self,
        node_idx: usize,
        query: &[F],
        k: usize,
        heap: &mut std::collections::BinaryHeap<(OrderedFloat<F>, usize)>,
        search_radius: &mut F,
    ) {
        let node = &self.nodes[node_idx];

        // Calculate distance from query to ball center
        let center_dist = euclidean_distance(query, &node.center);

        // Enhanced distance bounds checking
        let min_possible_dist = if center_dist > node.radius {
            center_dist - node.radius
        } else {
            F::zero()
        };

        // Get current kth distance for pruning
        let kth_dist = if heap.len() < k {
            *search_radius
        } else {
            match heap.peek() {
                Some(&(dist, _)) => dist.into_inner(),
                None => *search_radius,
            }
        };

        // Early pruning: if the closest possible point in this ball is farther than kth distance
        if min_possible_dist > kth_dist {
            return;
        }

        // If this is a leaf node, check all points
        if node.left.is_none() && node.right.is_none() {
            for &idx in &node.indices {
                let point = self.points.row(idx);
                let dist = euclidean_distance(query, &point.to_vec());

                // Only consider points within search radius
                if dist <= *search_radius {
                    heap.push((OrderedFloat(dist), idx));

                    // If heap is too large, remove the farthest point and update search radius
                    if heap.len() > k {
                        heap.pop();
                    }

                    // Update search radius to the farthest point in current k-nearest set
                    if heap.len() == k {
                        if let Some(&(max_dist, _)) = heap.peek() {
                            *search_radius = max_dist.into_inner();
                        }
                    }
                }
            }
            return;
        }

        // Process children with improved ordering strategy
        let left_idx = node.left.unwrap();
        let right_idx = node.right.unwrap();

        let left_node = &self.nodes[left_idx];
        let right_node = &self.nodes[right_idx];

        // Calculate minimum possible distances to each child ball
        let left_center_dist = euclidean_distance(query, &left_node.center);
        let right_center_dist = euclidean_distance(query, &right_node.center);

        let left_min_dist = if left_center_dist > left_node.radius {
            left_center_dist - left_node.radius
        } else {
            F::zero()
        };

        let right_min_dist = if right_center_dist > right_node.radius {
            right_center_dist - right_node.radius
        } else {
            F::zero()
        };

        // Order children by minimum possible distance
        let (first_idx, second_idx, second_min_dist) = if left_min_dist < right_min_dist {
            (left_idx, right_idx, right_min_dist)
        } else {
            (right_idx, left_idx, left_min_dist)
        };

        // Search the closer child first
        self.search_k_nearest_optimized(first_idx, query, k, heap, search_radius);

        // Re-check pruning condition for second child after first search
        let updated_kth_dist = if heap.len() < k {
            *search_radius
        } else {
            match heap.peek() {
                Some(&(dist, _)) => dist.into_inner(),
                None => *search_radius,
            }
        };

        // Only search second child if it could contain better points
        if second_min_dist <= updated_kth_dist {
            self.search_k_nearest_optimized(second_idx, query, k, heap, search_radius);
        }
    }

    /// Approximate nearest neighbor search with controlled accuracy
    ///
    /// This method provides faster approximate nearest neighbor search by
    /// exploring only a limited number of branches in the tree.
    ///
    /// # Arguments
    ///
    /// * `query` - Query point coordinates
    /// * `k` - Number of nearest neighbors to find
    /// * `max_checks` - Maximum number of distance computations
    ///
    /// # Returns
    ///
    /// Vector of (point_index, distance) tuples, sorted by distance
    pub fn approximate_k_nearest_neighbors(
        &self,
        query: &[F],
        k: usize,
        max_checks: usize,
    ) -> InterpolateResult<Vec<(usize, F)>> {
        // Check query dimension
        if query.len() != self.dim {
            return Err(InterpolateError::DimensionMismatch(format!(
                "Query dimension {} doesn't match Ball Tree dimension {}",
                query.len(),
                self.dim
            )));
        }

        // Handle empty tree
        if self.root.is_none() {
            return Err(InterpolateError::InvalidState(
                "Ball Tree is empty".to_string(),
            ));
        }

        // Limit k to the number of points
        let k = k.min(self.points.shape()[0]);

        if k == 0 {
            return Ok(Vec::new());
        }

        // For very small trees or large max_checks, use exact search
        if self.points.shape()[0] <= self.leaf_size || max_checks >= self.points.shape()[0] {
            return self.k_nearest_neighbors(query, k);
        }

        use std::collections::{BinaryHeap, VecDeque};

        let mut heap = BinaryHeap::with_capacity(k + 1);
        let mut checks_performed = 0;
        let mut nodes_to_visit = VecDeque::new();

        // Start with root
        nodes_to_visit.push_back((self.root.unwrap(), F::zero()));

        while let Some((node_idx, _min_dist)) = nodes_to_visit.pop_front() {
            if checks_performed >= max_checks {
                break;
            }

            let node = &self.nodes[node_idx];

            // Calculate distance from query to ball center
            let _center_dist = euclidean_distance(query, &node.center);

            // If this is a leaf node, check all points
            if node.left.is_none() && node.right.is_none() {
                for &idx in &node.indices {
                    if checks_performed >= max_checks {
                        break;
                    }

                    let point = self.points.row(idx);
                    let dist = euclidean_distance(query, &point.to_vec());
                    checks_performed += 1;

                    heap.push((OrderedFloat(dist), idx));

                    // If heap is too large, remove the farthest point
                    if heap.len() > k {
                        heap.pop();
                    }
                }
            } else {
                // Add children to queue, prioritizing by distance
                if let Some(left_idx) = node.left {
                    let left_node = &self.nodes[left_idx];
                    let left_center_dist = euclidean_distance(query, &left_node.center);
                    let left_min_dist = if left_center_dist > left_node.radius {
                        left_center_dist - left_node.radius
                    } else {
                        F::zero()
                    };

                    nodes_to_visit.push_back((left_idx, left_min_dist));
                }

                if let Some(right_idx) = node.right {
                    let right_node = &self.nodes[right_idx];
                    let right_center_dist = euclidean_distance(query, &right_node.center);
                    let right_min_dist = if right_center_dist > right_node.radius {
                        right_center_dist - right_node.radius
                    } else {
                        F::zero()
                    };

                    nodes_to_visit.push_back((right_idx, right_min_dist));
                }

                // Sort queue by minimum distance to prioritize closer nodes
                nodes_to_visit
                    .make_contiguous()
                    .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            }
        }

        // Convert heap to sorted vector
        let mut results: Vec<(usize, F)> = heap
            .into_iter()
            .map(|(dist, idx)| (idx, dist.into_inner()))
            .collect();

        // Sort by distance
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        Ok(results)
    }
}

/// Compute the centroid (center) of a set of points
fn compute_centroid<F: Float + FromPrimitive>(points: &Array2<F>, indices: &[usize]) -> Vec<F> {
    let n_points = indices.len();
    let n_dims = points.shape()[1];

    if n_points == 0 {
        return vec![F::zero(); n_dims];
    }

    let mut center = vec![F::zero(); n_dims];

    // Sum all point coordinates
    for &idx in indices {
        let point = points.row(idx);
        for d in 0..n_dims {
            center[d] = center[d] + point[d];
        }
    }

    // Divide by number of points
    let n = F::from_usize(n_points).unwrap();
    for val in center.iter_mut() {
        *val = *val / n;
    }

    center
}

/// Compute the radius of a ball containing all points
fn compute_radius<F: Float>(points: &Array2<F>, indices: &[usize], center: &[F]) -> F {
    let n_points = indices.len();

    if n_points == 0 {
        return F::zero();
    }

    let mut max_dist = F::zero();

    // Find the maximum distance from center to any point
    for &idx in indices {
        let point = points.row(idx);
        let dist = euclidean_distance(&point.to_vec(), center);

        if dist > max_dist {
            max_dist = dist;
        }
    }

    max_dist
}

/// Find the dimension with the largest spread of values
fn find_max_spread_dimension<F: Float>(points: &Array2<F>, indices: &[usize]) -> (usize, F) {
    let n_points = indices.len();
    let n_dims = points.shape()[1];

    if n_points <= 1 {
        return (0, F::zero());
    }

    let mut max_dim = 0;
    let mut max_spread = F::neg_infinity();

    // For each dimension, find the range of values
    for d in 0..n_dims {
        let mut min_val = F::infinity();
        let mut max_val = F::neg_infinity();

        for &idx in indices {
            let val = points[[idx, d]];

            if val < min_val {
                min_val = val;
            }

            if val > max_val {
                max_val = val;
            }
        }

        let spread = max_val - min_val;

        if spread > max_spread {
            max_spread = spread;
            max_dim = d;
        }
    }

    (max_dim, max_spread)
}

/// Find two points that are far apart along a given dimension
fn find_distant_points<F: Float>(
    points: &Array2<F>,
    indices: &[usize],
    dim: usize,
) -> (usize, usize) {
    let n_points = indices.len();

    if n_points <= 1 {
        return (indices[0], indices[0]);
    }

    // Find min and max points along the dimension
    let mut min_idx = indices[0];
    let mut max_idx = indices[0];
    let mut min_val = points[[min_idx, dim]];
    let mut max_val = min_val;

    for &idx in indices.iter().skip(1) {
        let val = points[[idx, dim]];

        if val < min_val {
            min_val = val;
            min_idx = idx;
        }

        if val > max_val {
            max_val = val;
            max_idx = idx;
        }
    }

    (min_idx, max_idx)
}

/// Partition points based on which of two seed points they're closer to
fn partition_by_seeds<F: Float>(
    points: &Array2<F>,
    indices: &[usize],
    seed1: usize,
    seed2: usize,
) -> (Vec<usize>, Vec<usize>) {
    let seed1_point = points.row(seed1).to_vec();
    let seed2_point = points.row(seed2).to_vec();

    let mut left_indices = Vec::new();
    let mut right_indices = Vec::new();

    // Always include the seeds in their respective partitions
    left_indices.push(seed1);
    right_indices.push(seed2);

    // Partition the remaining points
    for &idx in indices {
        if idx == seed1 || idx == seed2 {
            continue; // Skip the seeds
        }

        let point = points.row(idx).to_vec();
        let dist1 = euclidean_distance(&point, &seed1_point);
        let dist2 = euclidean_distance(&point, &seed2_point);

        if dist1 <= dist2 {
            left_indices.push(idx);
        } else {
            right_indices.push(idx);
        }
    }

    // If one partition is empty, move some points from the other
    if left_indices.is_empty() && right_indices.len() >= 2 {
        left_indices.push(right_indices.pop().unwrap());
    } else if right_indices.is_empty() && left_indices.len() >= 2 {
        right_indices.push(left_indices.pop().unwrap());
    }

    (left_indices, right_indices)
}

/// Calculate Euclidean distance between two points
fn euclidean_distance<F: Float>(a: &[F], b: &[F]) -> F {
    debug_assert_eq!(a.len(), b.len());

    let mut sum_sq = F::zero();

    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum_sq = sum_sq + diff * diff;
    }

    sum_sq.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_balltree_creation() {
        // Create a simple 3D dataset
        let points = arr2(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5],
        ]);

        let balltree = BallTree::new(points).unwrap();

        // Check tree properties
        assert_eq!(balltree.len(), 5);
        assert_eq!(balltree.dim(), 3);
        assert!(!balltree.is_empty());
    }

    #[test]
    fn test_nearest_neighbor() {
        // Create a simple 3D dataset
        let points = arr2(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5],
        ]);

        let balltree = BallTree::new(points).unwrap();

        // Test exact matches
        for i in 0..5 {
            let point = balltree.points().row(i).to_vec();
            let (idx, dist) = balltree.nearest_neighbor(&point).unwrap();
            assert_eq!(idx, i);
            assert!(dist < 1e-10);
        }

        // Test near matches
        let query = vec![0.6, 0.6, 0.6];
        let (idx, _) = balltree.nearest_neighbor(&query).unwrap();
        assert_eq!(idx, 4); // Should be closest to (0.5, 0.5, 0.5)

        let query = vec![0.9, 0.1, 0.1];
        let (idx, _) = balltree.nearest_neighbor(&query).unwrap();
        assert_eq!(idx, 1); // Should be closest to (1.0, 0.0, 0.0)
    }

    #[test]
    fn test_k_nearest_neighbors() {
        // Create a simple 3D dataset
        let points = arr2(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5],
        ]);

        let balltree = BallTree::new(points).unwrap();

        // Test at point (0.6, 0.6, 0.6)
        let query = vec![0.6, 0.6, 0.6];

        // Get 3 nearest neighbors
        let neighbors = balltree.k_nearest_neighbors(&query, 3).unwrap();

        // Should include (0.5, 0.5, 0.5) as the closest
        assert_eq!(neighbors.len(), 3);
        assert_eq!(neighbors[0].0, 4); // (0.5, 0.5, 0.5) should be first
    }

    #[test]
    fn test_points_within_radius() {
        // Create a simple 3D dataset
        let points = arr2(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5],
        ]);

        let balltree = BallTree::new(points).unwrap();

        // Test at origin with radius 0.7
        let query = vec![0.0, 0.0, 0.0];
        let radius = 0.7;

        let results = balltree.points_within_radius(&query, radius).unwrap();

        // Should include the origin and possibly (0.5, 0.5, 0.5) depending on threshold
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 0); // Origin should be first
    }
}
