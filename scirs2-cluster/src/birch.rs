//! BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) clustering algorithm
//!
//! BIRCH is an incremental clustering algorithm for large datasets. It builds a CF-tree
//! (Clustering Feature tree) to summarize the data and then applies a global clustering
//! algorithm on the leaf nodes.

use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};
use crate::vq::euclidean_distance;

/// Clustering Feature for summarizing a cluster
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ClusteringFeature<F: Float> {
    /// Number of data points in the cluster
    n: usize,
    /// Linear sum of data points
    linear_sum: Array1<F>,
    /// Squared sum of data points
    squared_sum: F,
}

#[allow(dead_code)]
impl<F: Float + FromPrimitive + ScalarOperand> ClusteringFeature<F> {
    /// Create a new CF from a single data point
    fn new(_datapoint: ArrayView1<F>) -> Self {
        let squared_sum = _datapoint.dot(&_datapoint);
        Self {
            n: 1,
            linear_sum: _datapoint.to_owned(),
            squared_sum,
        }
    }

    /// Create an empty CF
    fn empty(_nfeatures: usize) -> Self {
        Self {
            n: 0,
            linear_sum: Array1::zeros(_nfeatures),
            squared_sum: F::zero(),
        }
    }

    /// Add another CF to this one
    fn add(&mut self, other: &Self) {
        self.n += other.n;
        self.linear_sum = &self.linear_sum + &other.linear_sum;
        self.squared_sum = self.squared_sum + other.squared_sum;
    }

    /// Merge with another CF and return a new CF
    fn merge(&self, other: &Self) -> Self {
        let mut result = self.clone();
        result.add(other);
        result
    }

    /// Calculate the centroid
    fn centroid(&self) -> Array1<F> {
        if self.n == 0 {
            Array1::zeros(self.linear_sum.len())
        } else {
            &self.linear_sum / F::from(self.n).unwrap()
        }
    }

    /// Calculate the radius (average distance from centroid)
    fn radius(&self) -> F {
        if self.n <= 1 {
            F::zero()
        } else {
            let centroid = self.centroid();
            let centroid_ss = centroid.dot(&centroid);
            let variance = (self.squared_sum - F::from(self.n).unwrap() * centroid_ss)
                / F::from(self.n).unwrap();
            variance.max(F::zero()).sqrt()
        }
    }

    /// Calculate the diameter (average pairwise distance)
    fn diameter(&self) -> F {
        if self.n <= 1 {
            F::zero()
        } else {
            let n_f = F::from(self.n).unwrap();
            let term = (n_f * self.squared_sum - self.linear_sum.dot(&self.linear_sum))
                / (n_f * (n_f - F::one()));
            term.max(F::zero()).sqrt() * F::from(2.0).unwrap()
        }
    }
}

/// Node in the CF-tree
#[derive(Debug)]
#[allow(dead_code)]
struct CFNode<F: Float> {
    /// Whether this is a leaf node
    is_leaf: bool,
    /// CFs stored in this node
    cfs: Vec<ClusteringFeature<F>>,
    /// Child nodes (only for non-leaf nodes)
    children: Vec<CFNode<F>>,
    /// Parent node reference (would need Rc<RefCell> in real implementation)
    parent_index: Option<usize>,
}

#[allow(dead_code)]
impl<F: Float + FromPrimitive + ScalarOperand> CFNode<F> {
    /// Create a new leaf node
    fn new_leaf() -> Self {
        Self {
            is_leaf: true,
            cfs: Vec::new(),
            children: Vec::new(),
            parent_index: None,
        }
    }

    /// Create a new non-leaf node
    fn new_non_leaf() -> Self {
        Self {
            is_leaf: false,
            cfs: Vec::new(),
            children: Vec::new(),
            parent_index: None,
        }
    }

    /// Get the CF that summarizes this entire node
    fn get_cf(&self) -> Result<ClusteringFeature<F>> {
        if self.cfs.is_empty() {
            return Err(ClusteringError::InvalidInput(
                "Node has no CFs - cannot compute summary".to_string(),
            ));
        }

        let mut result = self.cfs[0].clone();
        for cf in self.cfs.iter().skip(1) {
            result.add(cf);
        }
        Ok(result)
    }
}

/// BIRCH clustering algorithm options
#[derive(Debug, Clone)]
pub struct BirchOptions<F: Float> {
    /// Maximum number of CFs in a leaf node
    pub branching_factor: usize,
    /// Maximum radius of a subcluster
    pub threshold: F,
    /// Number of clusters to extract
    pub n_clusters: Option<usize>,
}

impl<F: Float + FromPrimitive> Default for BirchOptions<F> {
    fn default() -> Self {
        Self {
            branching_factor: 50,
            threshold: F::from(0.5).unwrap(),
            n_clusters: None,
        }
    }
}

/// BIRCH clustering algorithm
pub struct Birch<F: Float> {
    options: BirchOptions<F>,
    root: Option<Box<CFNode<F>>>,
    leaf_entries: Vec<ClusteringFeature<F>>,
    n_features: Option<usize>,
}

impl<F: Float + FromPrimitive + Debug + ScalarOperand> Birch<F> {
    /// Create a new BIRCH instance
    pub fn new(options: BirchOptions<F>) -> Self {
        Self {
            options,
            root: None,
            leaf_entries: Vec::new(),
            n_features: None,
        }
    }

    /// Fit the BIRCH model to data
    pub fn fit(&mut self, data: ArrayView2<F>) -> Result<()> {
        let n_samples = data.shape()[0];
        let n_features = data.shape()[1];

        if n_samples == 0 {
            return Err(ClusteringError::InvalidInput(
                "Input data is empty".to_string(),
            ));
        }

        self.n_features = Some(n_features);
        self.root = Some(Box::new(CFNode::new_leaf()));
        self.leaf_entries.clear();

        // Insert each data point into the tree
        for i in 0..n_samples {
            let point = data.slice(s![i, ..]);
            self.insert_point(point)?;
        }

        Ok(())
    }

    /// Insert a single point into the CF-tree
    fn insert_point(&mut self, point: ArrayView1<F>) -> Result<()> {
        let newcf = ClusteringFeature::new(point);

        if self.leaf_entries.is_empty() {
            // First point - create initial CF
            self.leaf_entries.push(newcf);
            return Ok(());
        }

        // Find the closest leaf entry
        let (_leaf_idx, cf_idx) = self.find_closest_leaf(&newcf)?;

        // Try to absorb the point into the closest CF
        if cf_idx < self.leaf_entries.len() {
            let closest_cf = &self.leaf_entries[cf_idx];

            // Create a temporary merged CF to check if it would satisfy the threshold
            let merged_cf = closest_cf.merge(&newcf);

            // Check if the merged CF would have an acceptable radius
            if merged_cf.radius() <= self.options.threshold {
                // Absorb the point into the existing CF
                self.leaf_entries[cf_idx] = merged_cf;
            } else {
                // Cannot absorb - check if we can add a new CF
                if self.leaf_entries.len() < self.options.branching_factor {
                    // Add as new CF
                    self.leaf_entries.push(newcf);
                } else {
                    // Need to split or merge existing CFs
                    // For now, replace the furthest CF or merge with closest
                    self.handle_overflow(newcf)?;
                }
            }
        }

        Ok(())
    }

    /// Handle overflow when branching factor is exceeded
    fn handle_overflow(&mut self, newcf: ClusteringFeature<F>) -> Result<()> {
        // Simple strategy: find the two closest CFs and merge them, then add the new CF
        if self.leaf_entries.len() < 2 {
            self.leaf_entries.push(newcf);
            return Ok(());
        }

        let mut min_distance = F::infinity();
        let mut merge_idx1 = 0;
        let mut merge_idx2 = 1;

        // Find the two closest CFs to merge
        for i in 0..self.leaf_entries.len() {
            for j in (i + 1)..self.leaf_entries.len() {
                let centroid1 = self.leaf_entries[i].centroid();
                let centroid2 = self.leaf_entries[j].centroid();

                let mut distance = F::zero();
                for k in 0..centroid1.len() {
                    let diff = centroid1[k] - centroid2[k];
                    distance = distance + diff * diff;
                }
                distance = distance.sqrt();

                if distance < min_distance {
                    min_distance = distance;
                    merge_idx1 = i;
                    merge_idx2 = j;
                }
            }
        }

        // Merge the two closest CFs
        let cf1 = self.leaf_entries[merge_idx1].clone();
        let cf2 = self.leaf_entries[merge_idx2].clone();
        let merged = cf1.merge(&cf2);

        // Remove the CFs being merged (remove higher index first to avoid index shifting)
        if merge_idx1 > merge_idx2 {
            self.leaf_entries.remove(merge_idx1);
            self.leaf_entries.remove(merge_idx2);
        } else {
            self.leaf_entries.remove(merge_idx2);
            self.leaf_entries.remove(merge_idx1);
        }

        // Add the merged CF and the new CF
        self.leaf_entries.push(merged);
        self.leaf_entries.push(newcf);

        Ok(())
    }

    /// Find the closest leaf entry to a CF
    fn find_closest_leaf(&self, cf: &ClusteringFeature<F>) -> Result<(usize, usize)> {
        if self.leaf_entries.is_empty() {
            return Err(ClusteringError::InvalidInput(
                "No leaf entries available".to_string(),
            ));
        }

        let mut min_distance = F::infinity();
        let mut closest_idx = 0;

        // Find the closest existing CF by comparing centroids
        for (i, existing_cf) in self.leaf_entries.iter().enumerate() {
            let centroid1 = cf.centroid();
            let centroid2 = existing_cf.centroid();

            // Calculate Euclidean distance between centroids
            let mut distance = F::zero();
            for k in 0..centroid1.len() {
                let diff = centroid1[k] - centroid2[k];
                distance = distance + diff * diff;
            }
            distance = distance.sqrt();

            if distance < min_distance {
                min_distance = distance;
                closest_idx = i;
            }
        }

        Ok((0, closest_idx)) // Return leaf index (0) and CF index within that leaf
    }

    /// Predict cluster labels for new data
    pub fn predict(&self, data: ArrayView2<F>) -> Result<Array1<i32>> {
        if self.leaf_entries.is_empty() {
            return Err(ClusteringError::InvalidInput(
                "Model has not been fitted yet".to_string(),
            ));
        }

        let n_samples = data.shape()[0];
        let mut labels = Array1::zeros(n_samples);

        // For each data point, find the closest leaf CF
        for i in 0..n_samples {
            let point = data.slice(s![i, ..]);
            let mut min_dist = F::infinity();
            let mut closest_cf = 0;

            // Compare with centroids of leaf CFs
            for (j, cf) in self.leaf_entries.iter().enumerate() {
                let centroid = cf.centroid();
                let dist = euclidean_distance(point, centroid.view());

                if dist < min_dist {
                    min_dist = dist;
                    closest_cf = j;
                }
            }

            labels[i] = closest_cf as i32;
        }

        Ok(labels)
    }

    /// Extract the final clusters from the CF-tree
    pub fn extract_clusters(&self) -> Result<(Array2<F>, Array1<i32>)> {
        if self.leaf_entries.is_empty() {
            return Err(ClusteringError::InvalidInput(
                "No data has been processed".to_string(),
            ));
        }

        let n_features = self.n_features.unwrap();
        let n_cf_entries = self.leaf_entries.len();
        let n_clusters = self.options.n_clusters.unwrap_or(n_cf_entries);

        if n_clusters >= n_cf_entries {
            // Use all CFs as clusters
            let mut centroids = Array2::zeros((n_cf_entries, n_features));
            let mut labels = Array1::zeros(n_cf_entries);

            for (i, cf) in self.leaf_entries.iter().enumerate() {
                let centroid = cf.centroid();
                centroids.slice_mut(s![i, ..]).assign(&centroid);
                labels[i] = i as i32;
            }

            Ok((centroids, labels))
        } else {
            // Apply further clustering on CF centroids
            self.cluster_cf_entries(n_clusters, n_features)
        }
    }

    /// Apply clustering on CF entries to reduce to desired number of clusters
    fn cluster_cf_entries(
        &self,
        n_clusters: usize,
        n_features: usize,
    ) -> Result<(Array2<F>, Array1<i32>)> {
        // Extract centroids from CFs
        let mut cf_centroids = Array2::zeros((self.leaf_entries.len(), n_features));
        for (i, cf) in self.leaf_entries.iter().enumerate() {
            let centroid = cf.centroid();
            cf_centroids.slice_mut(s![i, ..]).assign(&centroid);
        }

        // Apply simple k-means-like clustering on CF centroids
        let cluster_assignments = self.simple_kmeans_on_cfs(&cf_centroids, n_clusters)?;

        // Compute final cluster centroids weighted by CF sizes
        let mut final_centroids = Array2::zeros((n_clusters, n_features));
        let mut cluster_weights = vec![F::zero(); n_clusters];

        for (cf_idx, &cluster_id) in cluster_assignments.iter().enumerate() {
            let cf = &self.leaf_entries[cf_idx];
            let cf_centroid = cf.centroid();
            let cf_weight = F::from_usize(cf.n).unwrap();

            cluster_weights[cluster_id as usize] = cluster_weights[cluster_id as usize] + cf_weight;

            for j in 0..n_features {
                final_centroids[[cluster_id as usize, j]] =
                    final_centroids[[cluster_id as usize, j]] + cf_centroid[j] * cf_weight;
            }
        }

        // Normalize by weights
        for i in 0..n_clusters {
            if cluster_weights[i] > F::zero() {
                for j in 0..n_features {
                    final_centroids[[i, j]] = final_centroids[[i, j]] / cluster_weights[i];
                }
            }
        }

        Ok((final_centroids, cluster_assignments))
    }

    /// Simple k-means clustering on CF centroids
    fn simple_kmeans_on_cfs(&self, centroids: &Array2<F>, k: usize) -> Result<Array1<i32>> {
        let n_points = centroids.shape()[0];
        let n_features = centroids.shape()[1];

        if k >= n_points {
            // Each CF gets its own cluster
            return Ok(Array1::from_iter(0..(n_points as i32)));
        }

        // Initialize cluster centers by taking first k CFs
        let mut cluster_centers = Array2::zeros((k, n_features));
        for i in 0..k {
            for j in 0..n_features {
                cluster_centers[[i, j]] = centroids[[i % n_points, j]];
            }
        }

        let mut assignments = Array1::zeros(n_points);

        // Simple iteration - just one pass for efficiency
        for point_idx in 0..n_points {
            let mut min_dist = F::infinity();
            let mut closest_cluster = 0;

            for cluster_idx in 0..k {
                let mut dist = F::zero();
                for feature_idx in 0..n_features {
                    let diff = centroids[[point_idx, feature_idx]]
                        - cluster_centers[[cluster_idx, feature_idx]];
                    dist = dist + diff * diff;
                }
                dist = dist.sqrt();

                if dist < min_dist {
                    min_dist = dist;
                    closest_cluster = cluster_idx;
                }
            }

            assignments[point_idx] = closest_cluster as i32;
        }

        Ok(assignments)
    }

    /// Get statistics about the CF-tree
    pub fn get_statistics(&self) -> BirchStatistics<F> {
        let total_points: usize = self.leaf_entries.iter().map(|cf| cf.n).sum();
        let avg_cf_size = if !self.leaf_entries.is_empty() {
            total_points as f64 / self.leaf_entries.len() as f64
        } else {
            0.0
        };

        let avg_radius = if !self.leaf_entries.is_empty() {
            let total_radius: F = self
                .leaf_entries
                .iter()
                .map(|cf| cf.radius())
                .fold(F::zero(), |acc, x| acc + x);
            total_radius / F::from_usize(self.leaf_entries.len()).unwrap()
        } else {
            F::zero()
        };

        BirchStatistics {
            num_cf_entries: self.leaf_entries.len(),
            total_points,
            avg_cf_size,
            avg_radius,
            threshold: self.options.threshold,
            branching_factor: self.options.branching_factor,
        }
    }
}

/// Statistics about a BIRCH CF-tree
#[derive(Debug)]
pub struct BirchStatistics<F: Float> {
    /// Number of CF entries in the tree
    pub num_cf_entries: usize,
    /// Total number of data points processed
    pub total_points: usize,
    /// Average number of points per CF
    pub avg_cf_size: f64,
    /// Average radius of CFs
    pub avg_radius: F,
    /// Threshold parameter used
    pub threshold: F,
    /// Branching factor used
    pub branching_factor: usize,
}

/// BIRCH clustering convenience function
///
/// # Arguments
///
/// * `data` - Input data (n_samples × n_features)
/// * `options` - BIRCH options
///
/// # Returns
///
/// * Tuple of (centroids, labels)
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_cluster::birch::{birch, BirchOptions};
///
/// let data = Array2::from_shape_vec((6, 2), vec![
///     1.0, 2.0,
///     1.2, 1.8,
///     0.8, 1.9,
///     4.0, 5.0,
///     4.2, 4.8,
///     3.9, 5.1,
/// ]).unwrap();
///
/// let options = BirchOptions {
///     n_clusters: Some(2),
///     ..Default::default()
/// };
///
/// let (centroids, labels) = birch(data.view(), options).unwrap();
/// ```
#[allow(dead_code)]
pub fn birch<F>(data: ArrayView2<F>, options: BirchOptions<F>) -> Result<(Array2<F>, Array1<i32>)>
where
    F: Float + FromPrimitive + Debug + ScalarOperand,
{
    let mut model = Birch::new(options);
    model.fit(data)?;
    let (centroids, _cf_labels) = model.extract_clusters()?;
    let labels = model.predict(data)?;
    Ok((centroids, labels))
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_clustering_feature() {
        let point = Array1::from_vec(vec![1.0, 2.0]);
        let cf = ClusteringFeature::<f64>::new(point.view());

        assert_eq!(cf.n, 1);
        assert_eq!(cf.linear_sum, point);
        assert_eq!(cf.squared_sum, 5.0); // 1^2 + 2^2

        let centroid = cf.centroid();
        assert_eq!(centroid, point);
    }

    #[test]
    fn test_cf_merge() {
        let point1 = Array1::from_vec(vec![1.0, 2.0]);
        let point2 = Array1::from_vec(vec![3.0, 4.0]);

        let cf1 = ClusteringFeature::new(point1.view());
        let cf2 = ClusteringFeature::new(point2.view());

        let merged = cf1.merge(&cf2);

        assert_eq!(merged.n, 2);
        assert_eq!(merged.linear_sum, Array1::from_vec(vec![4.0, 6.0]));
        assert_eq!(merged.squared_sum, 30.0); // 1^2 + 2^2 + 3^2 + 4^2

        let centroid = merged.centroid();
        assert_eq!(centroid, Array1::from_vec(vec![2.0, 3.0]));
    }

    #[test]
    fn test_birch_simple() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.2, 1.8, 0.8, 1.9, 4.0, 5.0, 4.2, 4.8, 3.9, 5.1],
        )
        .unwrap();

        let options = BirchOptions {
            n_clusters: Some(2),
            threshold: 1.0,
            ..Default::default()
        };

        let result = birch(data.view(), options);
        assert!(result.is_ok());

        let (centroids, labels) = result.unwrap();
        assert_eq!(centroids.shape()[0], 2);
        assert_eq!(labels.len(), 6);
    }
}
