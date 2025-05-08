//! Ball tree for efficient nearest neighbor searches
//!
//! Ball trees are spatial data structures that organize points in a metric space into a tree structure.
//! Each node represents a hypersphere (ball) that contains a subset of the points.
//! This implementation shares similarities with KD-tree, but can be more efficient for high-dimensional data
//! or when using general distance metrics beyond Euclidean.
//!
//! ## Features
//!
//! * Fast construction of ball trees with customizable leaf size
//! * Nearest neighbor queries with configurable k
//! * Range queries to find all points within a distance
//! * Support for all distance metrics defined in the distance module
//! * Suitable for high-dimensional data where KD-trees become less efficient
//!
//! ## References
//!
//! * Omohundro, S.M. (1989) "Five Balltree Construction Algorithms"
//! * Liu, T. et al. (2006) "An Investigation of Practical Approximate Nearest Neighbor Algorithms"
//! * scikit-learn ball tree implementation

use crate::distance::{Distance, EuclideanDistance};
use crate::error::{SpatialError, SpatialResult};
use ndarray::{Array1, Array2, ArrayView2};
use num_traits::Float;
use std::cmp::Ordering;
use std::marker::PhantomData;

/// A node in the ball tree
#[derive(Clone, Debug)]
struct BallTreeNode<T: Float> {
    /// Index of the start of the points contained in this node
    start_idx: usize,

    /// Index of the end of the points contained in this node
    end_idx: usize,

    /// Centroid of the points in this node (center of the ball)
    centroid: Vec<T>,

    /// Radius of the ball that contains all points in this node
    radius: T,

    /// Index of the left child node
    left_child: Option<usize>,

    /// Index of the right child node
    right_child: Option<usize>,
}

/// Ball tree for efficient nearest neighbor searches
///
/// The ball tree partitions data into a set of nested hyperspheres (balls), which allows
/// for efficient nearest neighbor searches, especially in high-dimensional spaces.
/// Each node in the tree represents a ball containing a subset of the points.
///
/// # Type Parameters
///
/// * `T`: Floating point type (f32 or f64)
/// * `D`: Distance metric that implements the [`Distance`] trait
#[derive(Clone, Debug)]
pub struct BallTree<T: Float + Send + Sync, D: Distance<T>> {
    /// Points stored in the ball tree
    data: Array2<T>,

    /// Indices of points in the original array, reordered during tree construction
    indices: Array1<usize>,

    /// Nodes in the ball tree
    nodes: Vec<BallTreeNode<T>>,

    /// Number of data points
    n_samples: usize,

    /// Dimension of data points
    n_features: usize,

    /// Maximum number of points in leaf nodes
    leaf_size: usize,

    /// Distance metric to use
    distance: D,

    /// Phantom data for the float type
    _phantom: PhantomData<T>,
}

impl<T: Float + Send + Sync + 'static, D: Distance<T> + Send + Sync + 'static> BallTree<T, D> {
    /// Create a new ball tree from the given data points
    ///
    /// # Arguments
    ///
    /// * `data` - 2D array of data points (n_samples x n_features)
    /// * `leaf_size` - Maximum number of points in leaf nodes
    /// * `distance` - Distance metric to use
    ///
    /// # Returns
    ///
    /// * `SpatialResult<BallTree<T, D>>` - A new ball tree
    pub fn new(
        data: &ArrayView2<T>,
        leaf_size: usize,
        distance: D,
    ) -> SpatialResult<BallTree<T, D>> {
        let n_samples = data.nrows();
        let n_features = data.ncols();

        if n_samples == 0 {
            return Err(SpatialError::ValueError(
                "Input data array is empty".to_string(),
            ));
        }

        // Clone the data array and create an array of indices
        let data = data.to_owned();
        let indices = Array1::from_iter(0..n_samples);

        // Initialize empty nodes vector (will be filled during build)
        let nodes = Vec::new();

        let mut ball_tree = BallTree {
            data,
            indices,
            nodes,
            n_samples,
            n_features,
            leaf_size,
            distance,
            _phantom: PhantomData,
        };

        // Build the tree
        ball_tree.build_tree()?;

        Ok(ball_tree)
    }

    /// Build the ball tree recursively
    ///
    /// This initializes the tree structure and builds the nodes.
    fn build_tree(&mut self) -> SpatialResult<()> {
        if self.n_samples == 0 {
            return Ok(());
        }

        // Reserve space for the nodes (maximum nodes = 2*n_samples - 1)
        self.nodes = Vec::with_capacity(2 * self.n_samples);

        // Build the tree recursively
        self.build_subtree(0, self.n_samples)?;

        Ok(())
    }

    /// Build a subtree recursively
    ///
    /// # Arguments
    ///
    /// * `start_idx` - Start index of points for this subtree
    /// * `end_idx` - End index of points for this subtree
    ///
    /// # Returns
    ///
    /// * `SpatialResult<usize>` - Index of the root node of the subtree
    fn build_subtree(&mut self, start_idx: usize, end_idx: usize) -> SpatialResult<usize> {
        let n_points = end_idx - start_idx;

        // Calculate centroid of points in this node
        let mut centroid = vec![T::zero(); self.n_features];
        for i in start_idx..end_idx {
            let point_idx = self.indices[i];
            let point = self.data.row(point_idx);

            for (j, &val) in point.iter().take(self.n_features).enumerate() {
                centroid[j] = centroid[j] + val;
            }
        }

        for val in centroid.iter_mut().take(self.n_features) {
            *val = *val / T::from(n_points).unwrap();
        }

        // Calculate radius (maximum distance from centroid to any point)
        let mut radius = T::zero();
        for i in start_idx..end_idx {
            let point_idx = self.indices[i];
            let point = self.data.row(point_idx);

            let dist = self.distance.distance(&centroid, point.as_slice().unwrap());

            if dist > radius {
                radius = dist;
            }
        }

        // Create node
        let node_idx = self.nodes.len();
        let node = BallTreeNode {
            start_idx,
            end_idx,
            centroid,
            radius,
            left_child: None,
            right_child: None,
        };

        self.nodes.push(node);

        // If this is a leaf node (n_points <= leaf_size), we're done
        if n_points <= self.leaf_size {
            return Ok(node_idx);
        }

        // Otherwise, split the points and recursively build subtrees
        // We'll split along the direction of maximum variance
        self.split_points(node_idx, start_idx, end_idx)?;

        // Recursively build left and right subtrees
        let mid_idx = start_idx + n_points / 2;

        let left_idx = self.build_subtree(start_idx, mid_idx)?;
        let right_idx = self.build_subtree(mid_idx, end_idx)?;

        // Update node with child indices
        self.nodes[node_idx].left_child = Some(left_idx);
        self.nodes[node_idx].right_child = Some(right_idx);

        Ok(node_idx)
    }

    /// Split the points in a node into two groups
    ///
    /// This method partitions the points in a node into two groups,
    /// attempting to create a balanced split.
    ///
    /// # Arguments
    ///
    /// * `node_idx` - Index of the node to split
    /// * `start_idx` - Start index of points in the node
    /// * `end_idx` - End index of points in the node
    ///
    /// # Returns
    ///
    /// * `SpatialResult<()>` - Result of the split operation
    fn split_points(
        &mut self,
        node_idx: usize,
        start_idx: usize,
        end_idx: usize,
    ) -> SpatialResult<()> {
        // Find the dimension with the largest variance
        let node = &self.nodes[node_idx];
        let centroid = &node.centroid;

        // Calculate distances from centroid to all points
        let mut distances: Vec<(usize, T)> = (start_idx..end_idx)
            .map(|i| {
                let point_idx = self.indices[i];
                let point = self.data.row(point_idx);
                let dist = self.distance.distance(centroid, point.as_slice().unwrap());
                (i, dist)
            })
            .collect();

        // Sort points by distance from centroid
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        // Reorder indices array based on sorted distances
        // Midpoint is calculated but used implicitly when we reorder the indices
        let _mid_idx = start_idx + (end_idx - start_idx) / 2;
        let mut new_indices = Vec::with_capacity(end_idx - start_idx);

        for (i, _) in distances {
            new_indices.push(self.indices[i]);
        }

        for (i, idx) in new_indices.into_iter().enumerate() {
            self.indices[start_idx + i] = idx;
        }

        Ok(())
    }

    /// Query the k nearest neighbors to the given point
    ///
    /// # Arguments
    ///
    /// * `point` - Query point
    /// * `k` - Number of neighbors to find
    /// * `return_distance` - Whether to return distances
    ///
    /// # Returns
    ///
    /// * `SpatialResult<(Vec<usize>, Option<Vec<T>>)>` - Indices and optionally distances of the k nearest neighbors
    pub fn query(
        &self,
        point: &[T],
        k: usize,
        return_distance: bool,
    ) -> SpatialResult<(Vec<usize>, Option<Vec<T>>)> {
        if point.len() != self.n_features {
            return Err(SpatialError::DimensionError(format!(
                "Query point has {} dimensions, but tree has {} dimensions",
                point.len(),
                self.n_features
            )));
        }

        if k > self.n_samples {
            return Err(SpatialError::ValueError(format!(
                "k ({}) cannot be greater than the number of samples ({})",
                k, self.n_samples
            )));
        }

        // Store up to k nearest neighbors and their distances
        let mut nearest_neighbors = Vec::<(T, usize)>::with_capacity(k);
        let mut max_dist = T::infinity();

        // Perform the recursive search
        self.query_recursive(0, point, k, &mut nearest_neighbors, &mut max_dist);

        // Sort by distance
        nearest_neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        // Extract indices and distances
        let (distances, indices): (Vec<_>, Vec<_>) = nearest_neighbors.into_iter().unzip();

        // Return only distances if requested
        let distances_opt = if return_distance {
            Some(distances)
        } else {
            None
        };

        Ok((indices, distances_opt))
    }

    /// Recursively search for k nearest neighbors
    ///
    /// # Arguments
    ///
    /// * `node_idx` - Index of the current node
    /// * `point` - Query point
    /// * `k` - Number of neighbors to find
    /// * `nearest` - Vector of (distance, index) pairs for nearest neighbors
    /// * `max_dist` - Maximum distance to consider
    fn query_recursive(
        &self,
        node_idx: usize,
        point: &[T],
        k: usize,
        nearest: &mut Vec<(T, usize)>,
        max_dist: &mut T,
    ) {
        let node = &self.nodes[node_idx];

        // If this node is further than max_dist, skip it
        let dist_to_centroid = self.distance.distance(point, &node.centroid);
        if dist_to_centroid > node.radius + *max_dist {
            return;
        }

        // If this is a leaf node, check all points
        if node.left_child.is_none() {
            for i in node.start_idx..node.end_idx {
                let idx = self.indices[i];
                let dist = self
                    .distance
                    .distance(point, self.data.row(idx).as_slice().unwrap());

                if dist < *max_dist || nearest.len() < k {
                    // Add this point to nearest neighbors
                    nearest.push((dist, idx));

                    // If we have more than k points, remove the furthest
                    if nearest.len() > k {
                        // Find the index of the point with maximum distance
                        let max_idx = nearest
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| {
                                a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal)
                            })
                            .map(|(idx, _)| idx)
                            .unwrap();

                        // Remove that point
                        nearest.swap_remove(max_idx);

                        // Update max_dist to the new maximum distance
                        *max_dist = nearest
                            .iter()
                            .map(|(dist, _)| *dist)
                            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                            .unwrap_or(T::infinity());
                    }
                }
            }
            return;
        }

        // Otherwise, recursively search child nodes
        // Determine which child to search first (closest to the query point)
        let left_idx = node.left_child.unwrap();
        let right_idx = node.right_child.unwrap();

        let left_node = &self.nodes[left_idx];
        let right_node = &self.nodes[right_idx];

        let dist_left = self.distance.distance(point, &left_node.centroid);
        let dist_right = self.distance.distance(point, &right_node.centroid);

        // Search the closest child first
        if dist_left <= dist_right {
            self.query_recursive(left_idx, point, k, nearest, max_dist);
            self.query_recursive(right_idx, point, k, nearest, max_dist);
        } else {
            self.query_recursive(right_idx, point, k, nearest, max_dist);
            self.query_recursive(left_idx, point, k, nearest, max_dist);
        }
    }

    /// Find all points within a given radius of the query point
    ///
    /// # Arguments
    ///
    /// * `point` - Query point
    /// * `radius` - Radius to search within
    /// * `return_distance` - Whether to return distances
    ///
    /// # Returns
    ///
    /// * `SpatialResult<(Vec<usize>, Option<Vec<T>>)>` - Indices and optionally distances of points within radius
    pub fn query_radius(
        &self,
        point: &[T],
        radius: T,
        return_distance: bool,
    ) -> SpatialResult<(Vec<usize>, Option<Vec<T>>)> {
        if point.len() != self.n_features {
            return Err(SpatialError::DimensionError(format!(
                "Query point has {} dimensions, but tree has {} dimensions",
                point.len(),
                self.n_features
            )));
        }

        if radius < T::zero() {
            return Err(SpatialError::ValueError(
                "Radius must be non-negative".to_string(),
            ));
        }

        // Collect points within radius
        let mut result_indices = Vec::new();
        let mut result_distances = Vec::new();

        // Search the tree recursively
        self.query_radius_recursive(0, point, radius, &mut result_indices, &mut result_distances);

        // Sort by distance if needed
        if !result_indices.is_empty() {
            let mut idx_dist: Vec<(usize, T)> =
                result_indices.into_iter().zip(result_distances).collect();

            idx_dist.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

            let (indices, distances): (Vec<_>, Vec<_>) = idx_dist.into_iter().unzip();

            let distances_opt = if return_distance {
                Some(distances)
            } else {
                None
            };

            Ok((indices, distances_opt))
        } else {
            Ok((
                Vec::new(),
                if return_distance {
                    Some(Vec::new())
                } else {
                    None
                },
            ))
        }
    }

    /// Recursively find all points within radius
    fn query_radius_recursive(
        &self,
        node_idx: usize,
        point: &[T],
        radius: T,
        indices: &mut Vec<usize>,
        distances: &mut Vec<T>,
    ) {
        let node = &self.nodes[node_idx];

        // If this node is too far, skip it
        let dist_to_centroid = self.distance.distance(point, &node.centroid);
        if dist_to_centroid > node.radius + radius {
            return;
        }

        // If this is a leaf node, check all points
        if node.left_child.is_none() {
            for i in node.start_idx..node.end_idx {
                let idx = self.indices[i];
                let dist = self
                    .distance
                    .distance(point, self.data.row(idx).as_slice().unwrap());

                if dist <= radius {
                    indices.push(idx);
                    distances.push(dist);
                }
            }
            return;
        }

        // Otherwise, recursively search child nodes
        let left_idx = node.left_child.unwrap();
        let right_idx = node.right_child.unwrap();

        self.query_radius_recursive(left_idx, point, radius, indices, distances);
        self.query_radius_recursive(right_idx, point, radius, indices, distances);
    }

    /// Find all pairs of points from two trees that are within a given radius
    ///
    /// # Arguments
    ///
    /// * `other` - Another ball tree
    /// * `radius` - Radius to search within
    ///
    /// # Returns
    ///
    /// * `SpatialResult<Vec<(usize, usize)>>` - Pairs of indices (self_idx, other_idx) within radius
    pub fn query_radius_tree(
        &self,
        other: &BallTree<T, D>,
        radius: T,
    ) -> SpatialResult<Vec<(usize, usize)>> {
        if self.n_features != other.n_features {
            return Err(SpatialError::DimensionError(format!(
                "Trees have different dimensions: {} and {}",
                self.n_features, other.n_features
            )));
        }

        if radius < T::zero() {
            return Err(SpatialError::ValueError(
                "Radius must be non-negative".to_string(),
            ));
        }

        let mut pairs = Vec::new();

        self.query_radius_tree_recursive(0, other, 0, radius, &mut pairs);

        Ok(pairs)
    }

    /// Recursively find all pairs of points from two trees that are within radius
    fn query_radius_tree_recursive(
        &self,
        self_node_idx: usize,
        other: &BallTree<T, D>,
        other_node_idx: usize,
        radius: T,
        pairs: &mut Vec<(usize, usize)>,
    ) {
        let self_node = &self.nodes[self_node_idx];
        let other_node = &other.nodes[other_node_idx];

        // Calculate minimum distance between nodes
        let dist_between_centroids = self
            .distance
            .distance(&self_node.centroid, &other_node.centroid);

        // If the minimum distance between nodes is greater than radius, we can skip
        if dist_between_centroids > self_node.radius + other_node.radius + radius {
            return;
        }

        // If both are leaf nodes, check all point pairs
        if self_node.left_child.is_none() && other_node.left_child.is_none() {
            for i in self_node.start_idx..self_node.end_idx {
                let self_idx = self.indices[i];
                let self_point = self.data.row(self_idx);

                for j in other_node.start_idx..other_node.end_idx {
                    let other_idx = other.indices[j];
                    let other_point = other.data.row(other_idx);

                    let dist = self.distance.distance(
                        self_point.as_slice().unwrap(),
                        other_point.as_slice().unwrap(),
                    );

                    if dist <= radius {
                        pairs.push((self_idx, other_idx));
                    }
                }
            }
            return;
        }

        // Otherwise, recursively search child nodes
        // Split the node with more points
        if self_node.left_child.is_some()
            && (other_node.left_child.is_none()
                || (self_node.end_idx - self_node.start_idx)
                    > (other_node.end_idx - other_node.start_idx))
        {
            let left_idx = self_node.left_child.unwrap();
            let right_idx = self_node.right_child.unwrap();

            self.query_radius_tree_recursive(left_idx, other, other_node_idx, radius, pairs);
            self.query_radius_tree_recursive(right_idx, other, other_node_idx, radius, pairs);
        } else if other_node.left_child.is_some() {
            let left_idx = other_node.left_child.unwrap();
            let right_idx = other_node.right_child.unwrap();

            self.query_radius_tree_recursive(self_node_idx, other, left_idx, radius, pairs);
            self.query_radius_tree_recursive(self_node_idx, other, right_idx, radius, pairs);
        }
    }

    /// Get the original data points
    pub fn get_data(&self) -> &Array2<T> {
        &self.data
    }

    /// Get the number of data points
    pub fn get_n_samples(&self) -> usize {
        self.n_samples
    }

    /// Get the dimension of data points
    pub fn get_n_features(&self) -> usize {
        self.n_features
    }

    /// Get the leaf size
    pub fn get_leaf_size(&self) -> usize {
        self.leaf_size
    }
}

// Implement constructor with default distance metric (Euclidean)
impl<T: Float + Send + Sync + 'static> BallTree<T, EuclideanDistance<T>> {
    /// Create a new ball tree with default Euclidean distance metric
    ///
    /// # Arguments
    ///
    /// * `data` - 2D array of data points (n_samples x n_features)
    /// * `leaf_size` - Maximum number of points in leaf nodes
    ///
    /// # Returns
    ///
    /// * `SpatialResult<BallTree<T, EuclideanDistance<T>>>` - A new ball tree
    pub fn with_euclidean_distance(
        data: &ArrayView2<T>,
        leaf_size: usize,
    ) -> SpatialResult<BallTree<T, EuclideanDistance<T>>> {
        BallTree::new(data, leaf_size, EuclideanDistance::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::euclidean;
    use approx::assert_relative_eq;
    use ndarray::arr2;

    #[test]
    fn test_ball_tree_construction() {
        let data = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]);

        let tree = BallTree::with_euclidean_distance(&data.view(), 2).unwrap();

        assert_eq!(tree.get_n_samples(), 5);
        assert_eq!(tree.get_n_features(), 2);
        assert_eq!(tree.get_leaf_size(), 2);
    }

    #[test]
    fn test_ball_tree_nearest_neighbor() {
        let data = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]);

        let tree = BallTree::with_euclidean_distance(&data.view(), 2).unwrap();

        // Test 1-NN
        let (indices, distances) = tree.query(&[5.1, 5.9], 1, true).unwrap();
        assert_eq!(indices, vec![2]); // Index of [5.0, 6.0]
        assert!(distances.is_some());
        assert_relative_eq!(distances.unwrap()[0], euclidean(&[5.1, 5.9], &[5.0, 6.0]));

        // Test 3-NN
        let (indices, distances) = tree.query(&[5.1, 5.9], 3, true).unwrap();
        assert_eq!(indices.len(), 3);
        assert!(indices.contains(&2)); // Should contain index of [5.0, 6.0]
        assert!(distances.is_some());
        assert_eq!(distances.unwrap().len(), 3);

        // Test without distances
        let (indices, distances) = tree.query(&[5.1, 5.9], 1, false).unwrap();
        assert_eq!(indices, vec![2]);
        assert!(distances.is_none());
    }

    #[test]
    fn test_ball_tree_radius_search() {
        let data = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]);

        let tree = BallTree::with_euclidean_distance(&data.view(), 2).unwrap();

        // Search with small radius
        let (indices, _distances) = tree.query_radius(&[5.0, 6.0], 1.0, true).unwrap();
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 2); // Only [5.0, 6.0] itself should be within radius 1.0

        // Search with larger radius
        let (indices, _distances) = tree.query_radius(&[5.0, 6.0], 3.0, true).unwrap();
        assert!(indices.len() > 1); // Should include neighbors

        // Test without distances
        let (indices, distances) = tree.query_radius(&[5.0, 6.0], 3.0, false).unwrap();
        assert!(indices.len() > 1);
        assert!(distances.is_none());
    }

    #[test]
    fn test_ball_tree_dual_tree_search() {
        let data1 = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);

        let data2 = arr2(&[[2.0, 2.0], [4.0, 4.0], [6.0, 6.0]]);

        let tree1 = BallTree::with_euclidean_distance(&data1.view(), 2).unwrap();
        let tree2 = BallTree::with_euclidean_distance(&data2.view(), 2).unwrap();

        // Test dual tree search with small radius
        let pairs = tree1.query_radius_tree(&tree2, 1.5).unwrap();
        assert_eq!(pairs.len(), 3); // Each point in data1 should be close to its corresponding point in data2

        // Test dual tree search with large radius
        let pairs = tree1.query_radius_tree(&tree2, 10.0).unwrap();
        assert_eq!(pairs.len(), 9); // All pairs should be within radius 10.0
    }

    #[test]
    fn test_ball_tree_empty_input() {
        let data = arr2(&[[0.0f64; 2]; 0]);
        let result = BallTree::with_euclidean_distance(&data.view(), 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_ball_tree_dimension_mismatch() {
        let data = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);

        let tree = BallTree::with_euclidean_distance(&data.view(), 2).unwrap();

        // Query with wrong dimension
        let result = tree.query(&[1.0], 1, false);
        assert!(result.is_err());

        let result = tree.query_radius(&[1.0, 2.0, 3.0], 1.0, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_ball_tree_invalid_parameters() {
        let data = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);

        let tree = BallTree::with_euclidean_distance(&data.view(), 2).unwrap();

        // Query with k > n_samples
        let result = tree.query(&[1.0, 2.0], 4, false);
        assert!(result.is_err());

        // Query with negative radius
        let result = tree.query_radius(&[1.0, 2.0], -1.0, false);
        assert!(result.is_err());
    }
}
