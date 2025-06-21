//! Efficient neighbor search algorithms for clustering
//!
//! This module provides various algorithms for fast nearest neighbor search,
//! which are crucial for density-based clustering and other algorithms that
//! require neighborhood computations.

use ndarray::{Array2, ArrayView1, ArrayView2};
use num_traits::{Float, FromPrimitive};
use std::collections::BinaryHeap;
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};

/// Configuration for neighbor search algorithms
#[derive(Debug, Clone)]
pub struct NeighborSearchConfig {
    /// Algorithm to use for neighbor search
    pub algorithm: NeighborSearchAlgorithm,
    /// Leaf size for tree-based algorithms
    pub leaf_size: usize,
    /// Number of hash tables for LSH
    pub n_hash_tables: usize,
    /// Number of hash functions per table for LSH
    pub n_hash_functions: usize,
    /// Whether to use parallel processing
    pub parallel: bool,
}

impl Default for NeighborSearchConfig {
    fn default() -> Self {
        Self {
            algorithm: NeighborSearchAlgorithm::Auto,
            leaf_size: 30,
            n_hash_tables: 10,
            n_hash_functions: 4,
            parallel: true,
        }
    }
}

/// Available neighbor search algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeighborSearchAlgorithm {
    /// Automatically choose the best algorithm based on data characteristics
    Auto,
    /// Brute force search (exact, O(n²) time)
    BruteForce,
    /// KD-Tree search (good for low dimensions)
    KDTree,
    /// Ball Tree search (good for high dimensions)
    BallTree,
    /// Locality Sensitive Hashing (approximate, very fast)
    LSH,
}

/// Neighbor search result
#[derive(Debug, Clone)]
pub struct NeighborResult {
    /// Indices of the neighbors
    pub indices: Vec<usize>,
    /// Distances to the neighbors
    pub distances: Vec<f64>,
}

/// Trait for neighbor search implementations
pub trait NeighborSearcher<F: Float> {
    /// Build the search structure from data
    fn fit(&mut self, data: ArrayView2<F>) -> Result<()>;

    /// Find k nearest neighbors for a query point
    fn kneighbors(&self, query: ArrayView1<F>, k: usize) -> Result<NeighborResult>;

    /// Find all neighbors within radius
    fn radius_neighbors(&self, query: ArrayView1<F>, radius: F) -> Result<NeighborResult>;

    /// Find k nearest neighbors for multiple query points
    fn kneighbors_batch(&self, queries: ArrayView2<F>, k: usize) -> Result<Vec<NeighborResult>> {
        let mut results = Vec::new();
        for query in queries.outer_iter() {
            results.push(self.kneighbors(query, k)?);
        }
        Ok(results)
    }
}

/// KD-Tree implementation for fast nearest neighbor search
///
/// Works best for low-dimensional data (typically < 20 dimensions).
/// Uses spatial partitioning to achieve O(log n) average query time.
#[derive(Debug)]
pub struct KDTree<F: Float> {
    data: Option<Array2<F>>,
    tree: Option<KDNode>,
    leaf_size: usize,
}

#[derive(Debug, Clone)]
struct KDNode {
    /// Indices of points in this node
    indices: Vec<usize>,
    /// Splitting dimension
    split_dim: usize,
    /// Splitting value
    split_val: f64,
    /// Left child (points with split_dim < split_val)
    left: Option<Box<KDNode>>,
    /// Right child (points with split_dim >= split_val)
    right: Option<Box<KDNode>>,
    /// Whether this is a leaf node
    is_leaf: bool,
}

impl<F: Float + FromPrimitive + Debug> KDTree<F> {
    /// Create a new KD-Tree
    pub fn new(leaf_size: usize) -> Self {
        Self {
            data: None,
            tree: None,
            leaf_size,
        }
    }
}

impl<F: Float + FromPrimitive + Debug> NeighborSearcher<F> for KDTree<F> {
    fn fit(&mut self, data: ArrayView2<F>) -> Result<()> {
        let n_samples = data.shape()[0];
        let n_features = data.shape()[1];

        if n_samples == 0 {
            return Err(ClusteringError::InvalidInput(
                "Cannot fit on empty data".into(),
            ));
        }

        // Store the data
        self.data = Some(data.to_owned());

        // Build the tree
        let indices: Vec<usize> = (0..n_samples).collect();
        self.tree = Some(self.build_tree(indices, 0, n_features)?);

        Ok(())
    }

    fn kneighbors(&self, query: ArrayView1<F>, k: usize) -> Result<NeighborResult> {
        let data = self
            .data
            .as_ref()
            .ok_or_else(|| ClusteringError::InvalidInput("Tree not fitted yet".into()))?;

        let tree = self
            .tree
            .as_ref()
            .ok_or_else(|| ClusteringError::InvalidInput("Tree not built yet".into()))?;

        if k == 0 {
            return Ok(NeighborResult {
                indices: vec![],
                distances: vec![],
            });
        }

        let mut heap = BinaryHeap::new();
        self.search_knn(tree, query, k, data.view(), &mut heap);

        // Extract results from heap (in reverse order)
        let mut indices = Vec::new();
        let mut distances = Vec::new();

        while let Some(neighbor) = heap.pop() {
            indices.push(neighbor.index);
            distances.push(neighbor.distance);
        }

        // Reverse to get nearest first
        indices.reverse();
        distances.reverse();

        Ok(NeighborResult { indices, distances })
    }

    fn radius_neighbors(&self, query: ArrayView1<F>, radius: F) -> Result<NeighborResult> {
        let data = self
            .data
            .as_ref()
            .ok_or_else(|| ClusteringError::InvalidInput("Tree not fitted yet".into()))?;

        let tree = self
            .tree
            .as_ref()
            .ok_or_else(|| ClusteringError::InvalidInput("Tree not built yet".into()))?;

        let mut result = NeighborResult {
            indices: Vec::new(),
            distances: Vec::new(),
        };

        let radius_f64 = radius.to_f64().unwrap_or(0.0);
        self.search_radius(tree, query, radius_f64, data.view(), &mut result);

        Ok(result)
    }
}

impl<F: Float + FromPrimitive + Debug> KDTree<F> {
    fn build_tree(
        &self,
        mut indices: Vec<usize>,
        depth: usize,
        n_features: usize,
    ) -> Result<KDNode> {
        if indices.len() <= self.leaf_size {
            return Ok(KDNode {
                indices,
                split_dim: 0,
                split_val: 0.0,
                left: None,
                right: None,
                is_leaf: true,
            });
        }

        let data = self.data.as_ref().unwrap();

        // Choose splitting dimension (cycling through dimensions)
        let split_dim = depth % n_features;

        // Sort indices by the splitting dimension
        indices.sort_by(|&a, &b| {
            let val_a = data[[a, split_dim]].to_f64().unwrap_or(0.0);
            let val_b = data[[b, split_dim]].to_f64().unwrap_or(0.0);
            val_a
                .partial_cmp(&val_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Find median
        let median_idx = indices.len() / 2;
        let split_val = data[[indices[median_idx], split_dim]]
            .to_f64()
            .unwrap_or(0.0);

        // Split indices
        let left_indices = indices[..median_idx].to_vec();
        let right_indices = indices[median_idx..].to_vec();

        // Recursively build children
        let left = if !left_indices.is_empty() {
            Some(Box::new(self.build_tree(
                left_indices,
                depth + 1,
                n_features,
            )?))
        } else {
            None
        };

        let right = if !right_indices.is_empty() {
            Some(Box::new(self.build_tree(
                right_indices,
                depth + 1,
                n_features,
            )?))
        } else {
            None
        };

        Ok(KDNode {
            indices: vec![], // Internal nodes don't store indices
            split_dim,
            split_val,
            left,
            right,
            is_leaf: false,
        })
    }

    #[allow(clippy::only_used_in_recursion)]
    fn search_knn(
        &self,
        node: &KDNode,
        query: ArrayView1<F>,
        k: usize,
        data: ArrayView2<F>,
        heap: &mut BinaryHeap<NeighborCandidate>,
    ) {
        if node.is_leaf {
            // Check all points in this leaf
            for &idx in &node.indices {
                let dist = euclidean_distance(query, data.row(idx));

                if heap.len() < k {
                    heap.push(NeighborCandidate {
                        distance: dist,
                        index: idx,
                    });
                } else if let Some(top) = heap.peek() {
                    if dist < top.distance {
                        heap.pop();
                        heap.push(NeighborCandidate {
                            distance: dist,
                            index: idx,
                        });
                    }
                }
            }
        } else {
            // Determine which child to visit first
            let query_val = query[node.split_dim].to_f64().unwrap_or(0.0);
            let (first_child, second_child) = if query_val < node.split_val {
                (&node.left, &node.right)
            } else {
                (&node.right, &node.left)
            };

            // Search the first child
            if let Some(child) = first_child {
                self.search_knn(child, query, k, data, heap);
            }

            // Check if we need to search the second child
            let split_dist = (query_val - node.split_val).abs();
            if heap.len() < k || heap.peek().is_none_or(|top| split_dist < top.distance) {
                if let Some(child) = second_child {
                    self.search_knn(child, query, k, data, heap);
                }
            }
        }
    }

    #[allow(clippy::only_used_in_recursion)]
    fn search_radius(
        &self,
        node: &KDNode,
        query: ArrayView1<F>,
        radius: f64,
        data: ArrayView2<F>,
        result: &mut NeighborResult,
    ) {
        if node.is_leaf {
            // Check all points in this leaf
            for &idx in &node.indices {
                let dist = euclidean_distance(query, data.row(idx));

                if dist <= radius {
                    result.indices.push(idx);
                    result.distances.push(dist);
                }
            }
        } else {
            // Check if the splitting hyperplane intersects the query sphere
            let query_val = query[node.split_dim].to_f64().unwrap_or(0.0);
            let split_dist = (query_val - node.split_val).abs();

            // Search children that might contain neighbors
            if query_val < node.split_val {
                if let Some(child) = &node.left {
                    self.search_radius(child, query, radius, data, result);
                }
                if split_dist <= radius {
                    if let Some(child) = &node.right {
                        self.search_radius(child, query, radius, data, result);
                    }
                }
            } else {
                if let Some(child) = &node.right {
                    self.search_radius(child, query, radius, data, result);
                }
                if split_dist <= radius {
                    if let Some(child) = &node.left {
                        self.search_radius(child, query, radius, data, result);
                    }
                }
            }
        }
    }
}

/// Neighbor candidate for priority queue
#[derive(Debug, Clone, PartialEq)]
struct NeighborCandidate {
    distance: f64,
    index: usize,
}

impl Eq for NeighborCandidate {}

impl Ord for NeighborCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl PartialOrd for NeighborCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Brute force neighbor search (exact but slow)
///
/// Uses direct distance computation between all pairs of points.
/// Guaranteed to find exact neighbors but has O(n²) time complexity.
#[derive(Debug)]
pub struct BruteForceSearch<F: Float> {
    data: Option<Array2<F>>,
}

impl<F: Float + FromPrimitive + Debug> BruteForceSearch<F> {
    /// Create a new brute force searcher
    pub fn new() -> Self {
        Self { data: None }
    }
}

impl<F: Float + FromPrimitive + Debug> Default for BruteForceSearch<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + FromPrimitive + Debug> NeighborSearcher<F> for BruteForceSearch<F> {
    fn fit(&mut self, data: ArrayView2<F>) -> Result<()> {
        if data.shape()[0] == 0 {
            return Err(ClusteringError::InvalidInput(
                "Cannot fit on empty data".into(),
            ));
        }

        self.data = Some(data.to_owned());
        Ok(())
    }

    fn kneighbors(&self, query: ArrayView1<F>, k: usize) -> Result<NeighborResult> {
        let data = self
            .data
            .as_ref()
            .ok_or_else(|| ClusteringError::InvalidInput("Searcher not fitted yet".into()))?;

        if k == 0 {
            return Ok(NeighborResult {
                indices: vec![],
                distances: vec![],
            });
        }

        let n_samples = data.shape()[0];
        let k_actual = k.min(n_samples);

        // Calculate all distances
        let mut candidates: Vec<NeighborCandidate> = (0..n_samples)
            .map(|i| {
                let dist = euclidean_distance(query, data.row(i));
                NeighborCandidate {
                    distance: dist,
                    index: i,
                }
            })
            .collect();

        // Sort by distance and take k nearest
        candidates.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        candidates.truncate(k_actual);

        let indices = candidates.iter().map(|c| c.index).collect();
        let distances = candidates.iter().map(|c| c.distance).collect();

        Ok(NeighborResult { indices, distances })
    }

    fn radius_neighbors(&self, query: ArrayView1<F>, radius: F) -> Result<NeighborResult> {
        let data = self
            .data
            .as_ref()
            .ok_or_else(|| ClusteringError::InvalidInput("Searcher not fitted yet".into()))?;

        let radius_f64 = radius.to_f64().unwrap_or(0.0);
        let n_samples = data.shape()[0];

        let mut indices = Vec::new();
        let mut distances = Vec::new();

        for i in 0..n_samples {
            let dist = euclidean_distance(query, data.row(i));
            if dist <= radius_f64 {
                indices.push(i);
                distances.push(dist);
            }
        }

        Ok(NeighborResult { indices, distances })
    }
}

/// Ball Tree implementation for high-dimensional nearest neighbor search
///
/// More effective than KD-Tree for high-dimensional data.
/// Uses hypersphere partitioning instead of hyperplane partitioning.
#[derive(Debug)]
pub struct BallTree<F: Float> {
    data: Option<Array2<F>>,
    tree: Option<BallNode>,
    leaf_size: usize,
}

#[derive(Debug, Clone)]
struct BallNode {
    /// Center of the ball
    center: Vec<f64>,
    /// Radius of the ball
    radius: f64,
    /// Indices of points in this node
    indices: Vec<usize>,
    /// Left child
    left: Option<Box<BallNode>>,
    /// Right child
    right: Option<Box<BallNode>>,
    /// Whether this is a leaf node
    is_leaf: bool,
}

impl<F: Float + FromPrimitive + Debug> BallTree<F> {
    /// Create a new Ball Tree
    pub fn new(leaf_size: usize) -> Self {
        Self {
            data: None,
            tree: None,
            leaf_size,
        }
    }
}

impl<F: Float + FromPrimitive + Debug> NeighborSearcher<F> for BallTree<F> {
    fn fit(&mut self, data: ArrayView2<F>) -> Result<()> {
        let n_samples = data.shape()[0];

        if n_samples == 0 {
            return Err(ClusteringError::InvalidInput(
                "Cannot fit on empty data".into(),
            ));
        }

        self.data = Some(data.to_owned());

        let indices: Vec<usize> = (0..n_samples).collect();
        self.tree = Some(self.build_ball_tree(indices, data.view())?);

        Ok(())
    }

    fn kneighbors(&self, query: ArrayView1<F>, k: usize) -> Result<NeighborResult> {
        let data = self
            .data
            .as_ref()
            .ok_or_else(|| ClusteringError::InvalidInput("Tree not fitted yet".into()))?;

        let tree = self
            .tree
            .as_ref()
            .ok_or_else(|| ClusteringError::InvalidInput("Tree not built yet".into()))?;

        if k == 0 {
            return Ok(NeighborResult {
                indices: vec![],
                distances: vec![],
            });
        }

        let mut heap = BinaryHeap::new();
        self.search_ball_knn(tree, query, k, data.view(), &mut heap);

        let mut indices = Vec::new();
        let mut distances = Vec::new();

        while let Some(neighbor) = heap.pop() {
            indices.push(neighbor.index);
            distances.push(neighbor.distance);
        }

        indices.reverse();
        distances.reverse();

        Ok(NeighborResult { indices, distances })
    }

    fn radius_neighbors(&self, query: ArrayView1<F>, radius: F) -> Result<NeighborResult> {
        let data = self
            .data
            .as_ref()
            .ok_or_else(|| ClusteringError::InvalidInput("Tree not fitted yet".into()))?;

        let tree = self
            .tree
            .as_ref()
            .ok_or_else(|| ClusteringError::InvalidInput("Tree not built yet".into()))?;

        let mut result = NeighborResult {
            indices: Vec::new(),
            distances: Vec::new(),
        };

        let radius_f64 = radius.to_f64().unwrap_or(0.0);
        self.search_ball_radius(tree, query, radius_f64, data.view(), &mut result);

        Ok(result)
    }
}

impl<F: Float + FromPrimitive + Debug> BallTree<F> {
    fn build_ball_tree(&self, indices: Vec<usize>, data: ArrayView2<F>) -> Result<BallNode> {
        if indices.len() <= self.leaf_size {
            let (center, radius) = self.compute_ball(&indices, data);
            return Ok(BallNode {
                center,
                radius,
                indices,
                left: None,
                right: None,
                is_leaf: true,
            });
        }

        // Find the dimension with the largest spread
        let n_features = data.shape()[1];
        let mut best_dim = 0;
        let mut best_spread = 0.0;

        for dim in 0..n_features {
            let mut min_val = f64::INFINITY;
            let mut max_val = f64::NEG_INFINITY;

            for &idx in &indices {
                let val = data[[idx, dim]].to_f64().unwrap_or(0.0);
                min_val = min_val.min(val);
                max_val = max_val.max(val);
            }

            let spread = max_val - min_val;
            if spread > best_spread {
                best_spread = spread;
                best_dim = dim;
            }
        }

        // Sort indices by the best dimension
        let mut sorted_indices = indices;
        sorted_indices.sort_by(|&a, &b| {
            let val_a = data[[a, best_dim]].to_f64().unwrap_or(0.0);
            let val_b = data[[b, best_dim]].to_f64().unwrap_or(0.0);
            val_a
                .partial_cmp(&val_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Split at median
        let split_idx = sorted_indices.len() / 2;
        let left_indices = sorted_indices[..split_idx].to_vec();
        let right_indices = sorted_indices[split_idx..].to_vec();

        // Recursively build children
        let left = if !left_indices.is_empty() {
            Some(Box::new(self.build_ball_tree(left_indices, data)?))
        } else {
            None
        };

        let right = if !right_indices.is_empty() {
            Some(Box::new(self.build_ball_tree(right_indices, data)?))
        } else {
            None
        };

        // Compute ball for this node
        let (center, radius) = self.compute_ball(&sorted_indices, data);

        Ok(BallNode {
            center,
            radius,
            indices: vec![], // Internal nodes don't store indices
            left,
            right,
            is_leaf: false,
        })
    }

    fn compute_ball(&self, indices: &[usize], data: ArrayView2<F>) -> (Vec<f64>, f64) {
        if indices.is_empty() {
            return (vec![], 0.0);
        }

        let n_features = data.shape()[1];
        let mut center = vec![0.0; n_features];

        // Compute centroid
        for &idx in indices {
            for j in 0..n_features {
                center[j] += data[[idx, j]].to_f64().unwrap_or(0.0);
            }
        }

        let n_points = indices.len() as f64;
        for val in &mut center {
            *val /= n_points;
        }

        // Compute radius (maximum distance from center)
        let mut radius = 0.0;
        for &idx in indices {
            let mut dist_sq = 0.0;
            for j in 0..n_features {
                let diff = data[[idx, j]].to_f64().unwrap_or(0.0) - center[j];
                dist_sq += diff * diff;
            }
            radius = radius.max(dist_sq.sqrt());
        }

        (center, radius)
    }

    #[allow(clippy::only_used_in_recursion)]
    fn search_ball_knn(
        &self,
        node: &BallNode,
        query: ArrayView1<F>,
        k: usize,
        data: ArrayView2<F>,
        heap: &mut BinaryHeap<NeighborCandidate>,
    ) {
        if node.is_leaf {
            // Check all points in this leaf
            for &idx in &node.indices {
                let dist = euclidean_distance(query, data.row(idx));

                if heap.len() < k {
                    heap.push(NeighborCandidate {
                        distance: dist,
                        index: idx,
                    });
                } else if let Some(top) = heap.peek() {
                    if dist < top.distance {
                        heap.pop();
                        heap.push(NeighborCandidate {
                            distance: dist,
                            index: idx,
                        });
                    }
                }
            }
        } else {
            // Check if this ball can contain better neighbors
            let query_vec: Vec<f64> = query.iter().map(|&x| x.to_f64().unwrap_or(0.0)).collect();

            let dist_to_center = euclidean_distance_vec(&query_vec, &node.center);
            let min_dist_to_ball = (dist_to_center - node.radius).max(0.0);

            if heap.len() < k
                || heap
                    .peek()
                    .is_none_or(|top| min_dist_to_ball < top.distance)
            {
                // Search children (closer one first)
                if let (Some(left), Some(right)) = (&node.left, &node.right) {
                    let left_dist = euclidean_distance_vec(&query_vec, &left.center);
                    let right_dist = euclidean_distance_vec(&query_vec, &right.center);

                    if left_dist <= right_dist {
                        self.search_ball_knn(left, query, k, data, heap);
                        self.search_ball_knn(right, query, k, data, heap);
                    } else {
                        self.search_ball_knn(right, query, k, data, heap);
                        self.search_ball_knn(left, query, k, data, heap);
                    }
                } else if let Some(child) = &node.left {
                    self.search_ball_knn(child, query, k, data, heap);
                } else if let Some(child) = &node.right {
                    self.search_ball_knn(child, query, k, data, heap);
                }
            }
        }
    }

    #[allow(clippy::only_used_in_recursion)]
    fn search_ball_radius(
        &self,
        node: &BallNode,
        query: ArrayView1<F>,
        radius: f64,
        data: ArrayView2<F>,
        result: &mut NeighborResult,
    ) {
        if node.is_leaf {
            // Check all points in this leaf
            for &idx in &node.indices {
                let dist = euclidean_distance(query, data.row(idx));

                if dist <= radius {
                    result.indices.push(idx);
                    result.distances.push(dist);
                }
            }
        } else {
            // Check if this ball intersects the query sphere
            let query_vec: Vec<f64> = query.iter().map(|&x| x.to_f64().unwrap_or(0.0)).collect();

            let dist_to_center = euclidean_distance_vec(&query_vec, &node.center);

            if dist_to_center <= radius + node.radius {
                // Ball intersects query sphere, search children
                if let Some(child) = &node.left {
                    self.search_ball_radius(child, query, radius, data, result);
                }
                if let Some(child) = &node.right {
                    self.search_ball_radius(child, query, radius, data, result);
                }
            }
        }
    }
}

/// Create the appropriate neighbor searcher based on configuration
pub fn create_neighbor_searcher<F: Float + FromPrimitive + Debug + 'static>(
    config: NeighborSearchConfig,
) -> Box<dyn NeighborSearcher<F>> {
    match config.algorithm {
        NeighborSearchAlgorithm::BruteForce => Box::new(BruteForceSearch::new()),
        NeighborSearchAlgorithm::KDTree => Box::new(KDTree::new(config.leaf_size)),
        NeighborSearchAlgorithm::BallTree => Box::new(BallTree::new(config.leaf_size)),
        NeighborSearchAlgorithm::Auto => {
            // Use KD-Tree by default (could be made smarter based on data characteristics)
            Box::new(KDTree::new(config.leaf_size))
        }
        NeighborSearchAlgorithm::LSH => {
            // LSH not implemented yet, fall back to KD-Tree
            Box::new(KDTree::new(config.leaf_size))
        }
    }
}

/// Calculate Euclidean distance between two points
fn euclidean_distance<F: Float + FromPrimitive>(a: ArrayView1<F>, b: ArrayView1<F>) -> f64 {
    let mut sum = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let diff = x.to_f64().unwrap_or(0.0) - y.to_f64().unwrap_or(0.0);
        sum += diff * diff;
    }
    sum.sqrt()
}

/// Calculate Euclidean distance between two f64 vectors
fn euclidean_distance_vec(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let diff = x - y;
        sum += diff * diff;
    }
    sum.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    fn create_test_data() -> Array2<f64> {
        Array2::from_shape_vec(
            (6, 2),
            vec![
                0.0, 0.0, // Point 0
                1.0, 0.0, // Point 1
                0.0, 1.0, // Point 2
                10.0, 10.0, // Point 3
                11.0, 10.0, // Point 4
                10.0, 11.0, // Point 5
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_brute_force_search() {
        let data = create_test_data();
        let mut searcher = BruteForceSearch::new();

        searcher.fit(data.view()).unwrap();

        // Query at origin - should find points 0, 1, 2 as nearest
        let query = Array1::from_vec(vec![0.0, 0.0]);
        let result = searcher.kneighbors(query.view(), 3).unwrap();

        assert_eq!(result.indices.len(), 3);
        assert_eq!(result.distances.len(), 3);

        // First neighbor should be point 0 (distance 0)
        assert_eq!(result.indices[0], 0);
        assert!(result.distances[0] < 1e-10);

        // Test radius search
        let radius_result = searcher.radius_neighbors(query.view(), 1.5).unwrap();
        assert!(radius_result.indices.len() >= 3); // Should find at least points 0, 1, 2
    }

    #[test]
    fn test_kdtree_search() {
        let data = create_test_data();
        let mut searcher = KDTree::new(2);

        searcher.fit(data.view()).unwrap();

        // Query at origin
        let query = Array1::from_vec(vec![0.0, 0.0]);
        let result = searcher.kneighbors(query.view(), 3).unwrap();

        assert_eq!(result.indices.len(), 3);
        assert_eq!(result.distances.len(), 3);

        // Should find the same nearest neighbors as brute force
        assert_eq!(result.indices[0], 0);
        assert!(result.distances[0] < 1e-10);
    }

    #[test]
    fn test_ball_tree_search() {
        let data = create_test_data();
        let mut searcher = BallTree::new(2);

        searcher.fit(data.view()).unwrap();

        // Query at origin
        let query = Array1::from_vec(vec![0.0, 0.0]);
        let result = searcher.kneighbors(query.view(), 3).unwrap();

        assert_eq!(result.indices.len(), 3);
        assert_eq!(result.distances.len(), 3);

        // Should find the same nearest neighbors as brute force
        assert_eq!(result.indices[0], 0);
        assert!(result.distances[0] < 1e-10);
    }

    #[test]
    fn test_neighbor_searcher_factory() {
        let data = create_test_data();

        let algorithms = vec![
            NeighborSearchAlgorithm::BruteForce,
            NeighborSearchAlgorithm::KDTree,
            NeighborSearchAlgorithm::BallTree,
            NeighborSearchAlgorithm::Auto,
        ];

        for algorithm in algorithms {
            let config = NeighborSearchConfig {
                algorithm,
                ..Default::default()
            };

            let mut searcher = create_neighbor_searcher(config);
            searcher.fit(data.view()).unwrap();

            let query = Array1::from_vec(vec![0.0, 0.0]);
            let result = searcher.kneighbors(query.view(), 2).unwrap();

            assert_eq!(result.indices.len(), 2);
            assert_eq!(result.distances.len(), 2);
        }
    }

    #[test]
    fn test_empty_data_error() {
        let empty_data: Array2<f64> = Array2::zeros((0, 2));
        let mut searcher = BruteForceSearch::new();

        let result = searcher.fit(empty_data.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_k_zero() {
        let data = create_test_data();
        let mut searcher = BruteForceSearch::new();
        searcher.fit(data.view()).unwrap();

        let query = Array1::from_vec(vec![0.0, 0.0]);
        let result = searcher.kneighbors(query.view(), 0).unwrap();

        assert_eq!(result.indices.len(), 0);
        assert_eq!(result.distances.len(), 0);
    }

    #[test]
    fn test_batch_queries() {
        let data = create_test_data();
        let mut searcher = BruteForceSearch::new();
        searcher.fit(data.view()).unwrap();

        let queries = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 10.0, 10.0]).unwrap();

        let results = searcher.kneighbors_batch(queries.view(), 2).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].indices.len(), 2);
        assert_eq!(results[1].indices.len(), 2);

        // First query should find point 0 first
        assert_eq!(results[0].indices[0], 0);
        // Second query should find point 3 first
        assert_eq!(results[1].indices[0], 3);
    }
}
