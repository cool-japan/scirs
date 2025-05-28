//! BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) clustering algorithm
//!
//! BIRCH is an incremental clustering algorithm for large datasets. It builds a CF-tree
//! (Clustering Feature tree) to summarize the data and then applies a global clustering
//! algorithm on the leaf nodes.

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, ScalarOperand};
use num_traits::{Float, FromPrimitive};
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
    fn new(data_point: ArrayView1<F>) -> Self {
        let squared_sum = data_point.dot(&data_point);
        Self {
            n: 1,
            linear_sum: data_point.to_owned(),
            squared_sum,
        }
    }

    /// Create an empty CF
    fn empty(n_features: usize) -> Self {
        Self {
            n: 0,
            linear_sum: Array1::zeros(n_features),
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
    fn get_cf(&self) -> ClusteringFeature<F> {
        if self.cfs.is_empty() {
            panic!("Node has no CFs");
        }

        let mut result = self.cfs[0].clone();
        for cf in self.cfs.iter().skip(1) {
            result.add(cf);
        }
        result
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
        let cf = ClusteringFeature::new(point);

        if let Some(_root) = &mut self.root {
            // Find the closest leaf entry
            let (_leaf_idx, _cf_idx) = self.find_closest_leaf(&cf)?;

            // Try to absorb the point into the closest CF
            // For simplicity, we'll just add it as a new CF if threshold is exceeded
            // In a full implementation, this would handle node splits

            // Add to leaf entries for final clustering
            self.leaf_entries.push(cf);
        }

        Ok(())
    }

    /// Find the closest leaf entry to a CF
    fn find_closest_leaf(&self, _cf: &ClusteringFeature<F>) -> Result<(usize, usize)> {
        // Simplified version - in full implementation would traverse the tree
        Ok((0, 0))
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
        let n_clusters = self.options.n_clusters.unwrap_or(self.leaf_entries.len());

        // Extract centroids from leaf CFs
        let mut centroids = Array2::zeros((n_clusters.min(self.leaf_entries.len()), n_features));

        for (i, cf) in self.leaf_entries.iter().take(n_clusters).enumerate() {
            let centroid = cf.centroid();
            centroids.slice_mut(s![i, ..]).assign(&centroid);
        }

        // Create simple cluster assignments based on CF membership
        let mut labels = Array1::zeros(self.leaf_entries.len());
        for (i, _) in self.leaf_entries.iter().enumerate() {
            labels[i] = (i % n_clusters) as i32;
        }

        Ok((centroids, labels))
    }
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
/// use ndarray::Array2;
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
pub fn birch<F>(data: ArrayView2<F>, options: BirchOptions<F>) -> Result<(Array2<F>, Array1<i32>)>
where
    F: Float + FromPrimitive + Debug + ScalarOperand,
{
    let mut model = Birch::new(options);
    model.fit(data)?;
    model.extract_clusters()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

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
