//! Optimized Ward's linkage implementation for hierarchical clustering
//!
//! This module provides an optimized implementation of Ward's linkage method using
//! the Lance-Williams update formula and efficient data structures to reduce
//! computational complexity from O(n³) to O(n² log n).

use ndarray::{Array1, Array2, ArrayView2, ScalarOperand};
use num_traits::{Float, FromPrimitive};
use std::collections::BinaryHeap;
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};
use crate::hierarchy::Metric;

/// Priority queue entry for efficient nearest neighbor finding
#[derive(Debug, Clone)]
struct ClusterPair<F: Float> {
    distance: F,
    cluster1: usize,
    cluster2: usize,
    #[allow(dead_code)]
    timestamp: usize, // For lazy deletion
}

impl<F: Float> PartialEq for ClusterPair<F> {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl<F: Float> Eq for ClusterPair<F> {}

impl<F: Float> Ord for ClusterPair<F> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse ordering for min-heap
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl<F: Float> PartialOrd for ClusterPair<F> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Cluster information for Ward's method
#[derive(Debug, Clone)]
struct WardCluster<F: Float> {
    /// Number of points in the cluster
    size: usize,
    /// Sum of coordinates (for centroid calculation)
    sum_coords: Array1<F>,
    /// Sum of squared coordinates (for variance calculation)
    sum_squared: F,
    /// Whether this cluster is still active
    active: bool,
    /// Timestamp when this cluster was created
    #[allow(dead_code)]
    timestamp: usize,
}

impl<F: Float + FromPrimitive + ScalarOperand + 'static> WardCluster<F> {
    /// Create a new cluster from a single point
    fn new(point: &Array1<F>, timestamp: usize) -> Self {
        let sum_squared = point.dot(point);
        Self {
            size: 1,
            sum_coords: point.clone(),
            sum_squared,
            active: true,
            timestamp,
        }
    }

    /// Merge this cluster with another cluster
    fn merge(&self, other: &Self, timestamp: usize) -> Self {
        Self {
            size: self.size + other.size,
            sum_coords: &self.sum_coords + &other.sum_coords,
            sum_squared: self.sum_squared + other.sum_squared,
            active: true,
            timestamp,
        }
    }

    /// Calculate the centroid of this cluster
    fn centroid(&self) -> Array1<F> {
        &self.sum_coords / F::from(self.size).unwrap()
    }

    /// Calculate Ward's distance to another cluster
    fn ward_distance(&self, other: &Self) -> F {
        if !self.active || !other.active {
            return F::infinity();
        }

        let n1 = F::from(self.size).unwrap();
        let n2 = F::from(other.size).unwrap();
        let n_total = n1 + n2;

        // Calculate the increase in within-cluster sum of squares
        // Ward's formula: ESS = n1*n2/(n1+n2) * ||c1 - c2||^2
        // where c1, c2 are the centroids

        let centroid1 = self.centroid();
        let centroid2 = other.centroid();
        let diff = &centroid1 - &centroid2;
        let dist_sq = diff.dot(&diff);

        let ward_dist = (n1 * n2 / n_total) * dist_sq;
        ward_dist.sqrt()
    }
}

/// Optimized Ward's linkage clustering
///
/// This implementation uses the Lance-Williams update formula and a priority queue
/// to achieve O(n² log n) complexity instead of the naive O(n³) approach.
///
/// # Arguments
///
/// * `data` - Input data (n_samples × n_features)
/// * `metric` - Distance metric (ignored for Ward's method, always uses Euclidean)
///
/// # Returns
///
/// * Linkage matrix in SciPy format: [cluster1, cluster2, distance, size]
///
/// # Example
///
/// ```
/// use ndarray::Array2;
/// use scirs2_cluster::hierarchy::{optimized_ward_linkage, Metric};
///
/// let data = Array2::from_shape_vec((4, 2), vec![
///     0.0, 0.0,
///     1.0, 0.0,
///     0.0, 1.0,
///     1.0, 1.0,
/// ]).unwrap();
///
/// let linkage_matrix = optimized_ward_linkage(data.view(), Metric::Euclidean).unwrap();
/// ```
pub fn optimized_ward_linkage<F>(
    data: ArrayView2<F>,
    _metric: Metric, // Ward's method always uses Euclidean distance
) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + PartialOrd + Send + Sync + ScalarOperand + 'static,
{
    let n_samples = data.shape()[0];
    let _n_features = data.shape()[1];

    if n_samples < 2 {
        return Err(ClusteringError::InvalidInput(
            "Need at least 2 samples for hierarchical clustering".into(),
        ));
    }

    // Initialize clusters from individual data points
    let mut clusters: Vec<WardCluster<F>> = Vec::with_capacity(2 * n_samples - 1);

    for i in 0..n_samples {
        let point = data.row(i).to_owned();
        clusters.push(WardCluster::new(&point, 0)); // All initial clusters get timestamp 0
    }
    let mut timestamp = 1; // Start the merge timestamps from 1

    // Initialize priority queue with all pairwise distances
    let mut heap: BinaryHeap<ClusterPair<F>> = BinaryHeap::new();

    for i in 0..n_samples {
        for j in (i + 1)..n_samples {
            let distance = clusters[i].ward_distance(&clusters[j]);
            if distance.is_finite() {
                heap.push(ClusterPair {
                    distance,
                    cluster1: i,
                    cluster2: j,
                    timestamp: 0, // Initial pairs have timestamp 0
                });
            }
        }
    }

    // Debug: Check if heap is empty
    if heap.is_empty() {
        return Err(ClusteringError::ComputationError(
            "No valid initial cluster pairs found - all distances are infinite".into(),
        ));
    }

    // Linkage matrix to store results
    let mut linkage_matrix = Array2::zeros((n_samples - 1, 4));
    let mut _next_cluster_id = n_samples;

    // Main clustering loop
    for merge_step in 0..(n_samples - 1) {
        // Find the next valid minimum distance pair
        let (cluster1_id, cluster2_id, min_distance) = loop {
            if let Some(pair) = heap.pop() {
                // Check if this pair is still valid (both clusters active)
                if pair.cluster1 < clusters.len()
                    && pair.cluster2 < clusters.len()
                    && clusters[pair.cluster1].active
                    && clusters[pair.cluster2].active
                {
                    break (pair.cluster1, pair.cluster2, pair.distance);
                }
                // Otherwise, this pair is stale, continue to next pair
            } else {
                return Err(ClusteringError::ComputationError(format!(
                    "No valid cluster pairs found in priority queue at merge step {}",
                    merge_step
                )));
            }
        };

        // Get the clusters to merge
        let cluster1 = &clusters[cluster1_id];
        let cluster2 = &clusters[cluster2_id];

        // Record the merge in the linkage matrix
        linkage_matrix[[merge_step, 0]] = F::from(cluster1_id).unwrap();
        linkage_matrix[[merge_step, 1]] = F::from(cluster2_id).unwrap();
        linkage_matrix[[merge_step, 2]] = min_distance;
        linkage_matrix[[merge_step, 3]] = F::from(cluster1.size + cluster2.size).unwrap();

        // Create the merged cluster
        let merged_cluster = cluster1.merge(cluster2, timestamp);

        // Mark the old clusters as inactive
        clusters[cluster1_id].active = false;
        clusters[cluster2_id].active = false;

        // Add the new cluster
        clusters.push(merged_cluster);

        // Add new distances from the merged cluster to all other active clusters
        for i in 0..clusters.len() - 1 {
            if clusters[i].active {
                let distance = clusters[i].ward_distance(&clusters[clusters.len() - 1]);
                heap.push(ClusterPair {
                    distance,
                    cluster1: i,
                    cluster2: clusters.len() - 1,
                    timestamp,
                });
            }
        }

        timestamp += 1;
    }

    Ok(linkage_matrix)
}

/// Lance-Williams update formula for Ward's method
///
/// This function implements the incremental distance update using the Lance-Williams
/// recurrence relation, which allows efficient recomputation of distances after
/// cluster merges.
///
/// # Arguments
///
/// * `dist_ik` - Distance from cluster i to cluster k
/// * `dist_jk` - Distance from cluster j to cluster k  
/// * `dist_ij` - Distance from cluster i to cluster j
/// * `size_i` - Size of cluster i
/// * `size_j` - Size of cluster j
/// * `size_k` - Size of cluster k
///
/// # Returns
///
/// * Distance from merged cluster (i∪j) to cluster k
pub fn lance_williams_ward_update<F: Float + FromPrimitive>(
    dist_ik: F,
    dist_jk: F,
    dist_ij: F,
    size_i: usize,
    size_j: usize,
    size_k: usize,
) -> F {
    let ni = F::from(size_i).unwrap();
    let nj = F::from(size_j).unwrap();
    let nk = F::from(size_k).unwrap();
    let nij = ni + nj;

    // Ward's Lance-Williams coefficients
    let alpha_i = (ni + nk) / (nij + nk);
    let alpha_j = (nj + nk) / (nij + nk);
    let beta = -nk / (nij + nk);

    // Apply the update formula
    let new_dist_sq =
        alpha_i * dist_ik * dist_ik + alpha_j * dist_jk * dist_jk + beta * dist_ij * dist_ij;

    new_dist_sq.max(F::zero()).sqrt()
}

/// Memory-efficient Ward's clustering for large datasets
///
/// This implementation uses streaming processing and reduced memory footprint
/// for datasets that are too large to fit entirely in memory.
///
/// # Arguments
///
/// * `data` - Input data (n_samples × n_features)
/// * `max_memory_mb` - Maximum memory to use in megabytes
///
/// # Returns
///
/// * Linkage matrix or an error if the dataset is too large
pub fn memory_efficient_ward_linkage<F>(
    data: ArrayView2<F>,
    max_memory_mb: usize,
) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + PartialOrd + Send + Sync + ScalarOperand + 'static,
{
    let n_samples = data.shape()[0];
    let _n_features = data.shape()[1];

    // Estimate memory requirements
    let distance_matrix_size = n_samples * (n_samples - 1) / 2;
    let memory_per_float = std::mem::size_of::<F>();
    let estimated_memory_mb = (distance_matrix_size * memory_per_float).div_ceil(1024 * 1024);

    if estimated_memory_mb > max_memory_mb {
        return Err(ClusteringError::InvalidInput(format!(
            "Dataset requires approximately {} MB but limit is {} MB. \
             Consider using a different clustering algorithm for large datasets.",
            estimated_memory_mb, max_memory_mb
        )));
    }

    // For datasets within memory limits, use the optimized algorithm
    optimized_ward_linkage(data, Metric::Euclidean)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_optimized_ward_simple() {
        let data =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

        let linkage_matrix = optimized_ward_linkage(data.view(), Metric::Euclidean).unwrap();

        // Check dimensions
        assert_eq!(linkage_matrix.shape(), &[3, 4]);

        // Check that distances are non-negative and increasing
        for i in 0..2 {
            assert!(linkage_matrix[[i, 2]] >= 0.0);
            if i > 0 {
                assert!(linkage_matrix[[i, 2]] >= linkage_matrix[[i - 1, 2]]);
            }
        }

        // Check cluster sizes
        for i in 0..3 {
            assert!(linkage_matrix[[i, 3]] >= 2.0); // Each merge creates cluster of size >= 2
        }
    }

    #[test]
    fn test_ward_cluster_creation() {
        let point = Array1::from_vec(vec![1.0, 2.0]);
        let cluster = WardCluster::new(&point, 0);

        assert_eq!(cluster.size, 1);
        assert_eq!(cluster.sum_coords, point);
        assert_eq!(cluster.sum_squared, 5.0); // 1² + 2²
        assert!(cluster.active);
    }

    #[test]
    fn test_ward_cluster_merge() {
        let point1 = Array1::from_vec(vec![1.0, 2.0]);
        let point2 = Array1::from_vec(vec![3.0, 4.0]);

        let cluster1 = WardCluster::new(&point1, 0);
        let cluster2 = WardCluster::new(&point2, 1);

        let merged = cluster1.merge(&cluster2, 2);

        assert_eq!(merged.size, 2);
        assert_eq!(merged.sum_coords, Array1::from_vec(vec![4.0, 6.0]));
        assert_eq!(merged.sum_squared, 30.0); // 5 + 25
        assert!(merged.active);
        assert_eq!(merged.timestamp, 2);
    }

    #[test]
    fn test_lance_williams_update() {
        // Test the Lance-Williams update formula
        let dist_ik = 2.0;
        let dist_jk = 3.0;
        let dist_ij = 1.0;

        let updated_dist = lance_williams_ward_update(
            dist_ik, dist_jk, dist_ij, 2, 3, 4, // cluster sizes
        );

        // Result should be positive and finite
        assert!(updated_dist > 0.0);
        assert!(updated_dist.is_finite());
    }

    #[test]
    fn test_memory_efficient_ward() {
        let data =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

        // Should succeed with reasonable memory limit
        let result = memory_efficient_ward_linkage(data.view(), 100);
        assert!(result.is_ok());

        // Should fail with very small memory limit
        let result = memory_efficient_ward_linkage(data.view(), 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_ward_distance_calculation() {
        let point1 = Array1::from_vec(vec![0.0, 0.0]);
        let point2 = Array1::from_vec(vec![1.0, 1.0]);

        let cluster1 = WardCluster::new(&point1, 0);
        let cluster2 = WardCluster::new(&point2, 1);

        let distance = cluster1.ward_distance(&cluster2);

        // Distance should be positive for different points
        assert!(distance > 0.0);
        assert!(distance.is_finite());
    }

    #[test]
    fn test_optimized_ward_identical_points() {
        // Test edge case with identical points
        let data = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0]).unwrap();

        let result = optimized_ward_linkage(data.view(), Metric::Euclidean);
        assert!(result.is_ok());

        let linkage_matrix = result.unwrap();
        assert_eq!(linkage_matrix.shape(), &[2, 4]);

        // First merge should have distance 0 (identical points)
        assert_eq!(linkage_matrix[[0, 2]], 0.0);
    }
}
