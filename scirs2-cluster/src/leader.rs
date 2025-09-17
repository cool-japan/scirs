//! Leader algorithm implementation for clustering
//!
//! The Leader algorithm is a simple, single-pass clustering algorithm that
//! processes data points sequentially, creating clusters on-the-fly.

use crate::error::{ClusteringError, Result};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::Float;
use std::fmt::Debug;

/// Leader algorithm clustering
///
/// The Leader algorithm is a simple single-pass clustering method that:
/// 1. Takes the first data point as the first cluster leader
/// 2. For each subsequent point:
///    - If it's within threshold distance of an existing leader, assign it to that cluster
///    - Otherwise, make it a new leader
///
/// # Type Parameters
///
/// * `F` - Floating point type (f32 or f64)
///
/// # Arguments
///
/// * `data` - Input data matrix of shape (n_samples, n_features)
/// * `threshold` - Distance threshold for creating new clusters
/// * `metric` - Distance metric function
///
/// # Returns
///
/// * `leaders` - Array of cluster leaders
/// * `labels` - Cluster assignments for each data point
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_cluster::leader::{leader_clustering, euclidean_distance};
///
/// let data = array![
///     [1.0, 2.0],
///     [1.2, 1.8],
///     [5.0, 4.0],
///     [5.2, 4.1],
/// ];
///
/// let (leaders, labels) = leader_clustering(data.view(), 1.0, euclidean_distance).unwrap();
/// ```
#[allow(dead_code)]
pub fn leader_clustering<F, D>(
    data: ArrayView2<F>,
    threshold: F,
    metric: D,
) -> Result<(Array2<F>, Array1<usize>)>
where
    F: Float + Debug,
    D: Fn(ArrayView1<F>, ArrayView1<F>) -> F,
{
    if data.is_empty() {
        return Err(ClusteringError::InvalidInput(
            "Input data is empty".to_string(),
        ));
    }

    if threshold <= F::zero() {
        return Err(ClusteringError::InvalidInput(
            "Threshold must be positive".to_string(),
        ));
    }

    let n_samples = data.nrows();
    let n_features = data.ncols();

    let mut leaders: Vec<Array1<F>> = Vec::new();
    let mut labels = Array1::zeros(n_samples);

    // Process each data point
    for (i, sample) in data.rows().into_iter().enumerate() {
        let mut min_distance = F::infinity();
        let mut closest_leader = 0;

        // Find the closest leader
        for (j, leader) in leaders.iter().enumerate() {
            let distance = metric(sample, leader.view());
            if distance < min_distance {
                min_distance = distance;
                closest_leader = j;
            }
        }

        // Assign to existing cluster or create new one
        if leaders.is_empty() || min_distance > threshold {
            // Create new cluster
            leaders.push(sample.to_owned());
            let label_idx = leaders.len() - 1;
            labels[i] = label_idx;
        } else {
            // Assign to existing cluster
            labels[i] = closest_leader;
        }
    }

    // Convert leaders to Array2
    let n_leaders = leaders.len();
    let mut leaders_array = Array2::zeros((n_leaders, n_features));
    for (i, leader) in leaders.iter().enumerate() {
        leaders_array.row_mut(i).assign(leader);
    }

    Ok((leaders_array, labels))
}

/// Euclidean distance function
#[allow(dead_code)]
pub fn euclidean_distance<F: Float>(a: ArrayView1<F>, b: ArrayView1<F>) -> F {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x - *y) * (*x - *y))
        .fold(F::zero(), |acc, x| acc + x)
        .sqrt()
}

/// Manhattan distance function
#[allow(dead_code)]
pub fn manhattan_distance<F: Float>(a: ArrayView1<F>, b: ArrayView1<F>) -> F {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x - *y).abs())
        .fold(F::zero(), |acc, x| acc + x)
}

/// Leader algorithm with order-dependent results
///
/// This variant processes points in the order they appear, which can lead
/// to different results based on data ordering.
pub struct LeaderClustering<F: Float> {
    threshold: F,
    leaders: Vec<Array1<F>>,
}

impl<F: Float + Debug> LeaderClustering<F> {
    /// Create a new Leader clustering instance
    pub fn new(threshold: F) -> Result<Self> {
        if threshold <= F::zero() {
            return Err(ClusteringError::InvalidInput(
                "Threshold must be positive".to_string(),
            ));
        }

        Ok(Self {
            threshold,
            leaders: Vec::new(),
        })
    }

    /// Fit the model to data
    pub fn fit(&mut self, data: ArrayView2<F>) -> Result<()> {
        self.leaders.clear();

        for sample in data.rows() {
            let mut min_distance = F::infinity();

            // Find the closest leader
            for leader in &self.leaders {
                let distance = euclidean_distance(sample, leader.view());
                if distance < min_distance {
                    min_distance = distance;
                }
            }

            // Create new cluster if needed
            if self.leaders.is_empty() || min_distance > self.threshold {
                self.leaders.push(sample.to_owned());
            }
        }

        Ok(())
    }

    /// Predict cluster labels for data
    pub fn predict(&self, data: ArrayView2<F>) -> Result<Array1<usize>> {
        if self.leaders.is_empty() {
            return Err(ClusteringError::InvalidState(
                "Model has not been fitted yet".to_string(),
            ));
        }

        let n_samples = data.nrows();
        let mut labels = Array1::zeros(n_samples);

        for (i, sample) in data.rows().into_iter().enumerate() {
            let mut min_distance = F::infinity();
            let mut closest_leader = 0;

            for (j, leader) in self.leaders.iter().enumerate() {
                let distance = euclidean_distance(sample, leader.view());
                if distance < min_distance {
                    min_distance = distance;
                    closest_leader = j;
                }
            }

            labels[i] = closest_leader;
        }

        Ok(labels)
    }

    /// Fit the model and return predictions
    pub fn fit_predict(&mut self, data: ArrayView2<F>) -> Result<Array1<usize>> {
        self.fit(data)?;
        self.predict(data)
    }

    /// Get the cluster leaders
    pub fn get_leaders(&self) -> Array2<F> {
        if self.leaders.is_empty() {
            return Array2::zeros((0, 0));
        }

        let n_leaders = self.leaders.len();
        let n_features = self.leaders[0].len();
        let mut leaders_array = Array2::zeros((n_leaders, n_features));

        for (i, leader) in self.leaders.iter().enumerate() {
            leaders_array.row_mut(i).assign(leader);
        }

        leaders_array
    }

    /// Get the number of clusters
    pub fn n_clusters(&self) -> usize {
        self.leaders.len()
    }
}

/// Tree representation for hierarchical organization of leaders
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LeaderTree<F: Float> {
    /// Root nodes of the tree
    pub roots: Vec<LeaderNode<F>>,
    /// Distance threshold for this level
    pub threshold: F,
}

/// Node in the leader tree structure
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LeaderNode<F: Float> {
    /// The leader vector
    pub leader: Array1<F>,
    /// Child nodes
    pub children: Vec<LeaderNode<F>>,
    /// Indices of data points in this cluster
    pub members: Vec<usize>,
}

impl<F: Float + Debug> LeaderTree<F> {
    /// Build a hierarchical leader tree with multiple threshold levels
    pub fn build_hierarchical(data: ArrayView2<F>, thresholds: &[F]) -> Result<Self> {
        if thresholds.is_empty() {
            return Err(ClusteringError::InvalidInput(
                "At least one threshold is required".to_string(),
            ));
        }

        // Start with the largest threshold
        let current_threshold = thresholds[0];
        let (leaders, labels) = leader_clustering(data, current_threshold, euclidean_distance)?;

        // Build root nodes
        let mut roots = Vec::new();
        for i in 0..leaders.nrows() {
            let mut members = Vec::new();
            for (j, &label) in labels.iter().enumerate() {
                if label == i {
                    members.push(j);
                }
            }

            roots.push(LeaderNode {
                leader: leaders.row(i).to_owned(),
                children: Vec::new(),
                members,
            });
        }

        // Build lower levels if more thresholds provided
        if thresholds.len() > 1 {
            for root in &mut roots {
                Self::build_subtree(data, root, &thresholds[1..])?;
            }
        }

        Ok(LeaderTree {
            roots,
            threshold: current_threshold,
        })
    }

    fn build_subtree(
        data: ArrayView2<F>,
        parent: &mut LeaderNode<F>,
        thresholds: &[F],
    ) -> Result<()> {
        if thresholds.is_empty() || parent.members.len() <= 1 {
            return Ok(());
        }

        // Extract data for this cluster
        let n_features = data.ncols();
        let mut cluster_data = Array2::zeros((parent.members.len(), n_features));
        for (i, &idx) in parent.members.iter().enumerate() {
            cluster_data.row_mut(i).assign(&data.row(idx));
        }

        // Cluster with smaller threshold
        let (sub_leaders, sub_labels) =
            leader_clustering(cluster_data.view(), thresholds[0], euclidean_distance)?;

        // Build child nodes
        for i in 0..sub_leaders.nrows() {
            let mut members = Vec::new();
            for (j, &label) in sub_labels.iter().enumerate() {
                if label == i {
                    members.push(parent.members[j]);
                }
            }

            let mut child = LeaderNode {
                leader: sub_leaders.row(i).to_owned(),
                children: Vec::new(),
                members,
            };

            // Recursively build subtree
            if thresholds.len() > 1 {
                Self::build_subtree(data, &mut child, &thresholds[1..])?;
            }

            parent.children.push(child);
        }

        Ok(())
    }

    /// Get the total number of nodes in the tree
    pub fn node_count(&self) -> usize {
        self.roots.iter().map(|root| Self::count_nodes(root)).sum()
    }

    fn count_nodes(node: &LeaderNode<F>) -> usize {
        1 + node
            .children
            .iter()
            .map(|child| Self::count_nodes(child))
            .sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_leader_clustering_basic() {
        let data = array![[1.0, 2.0], [1.2, 1.8], [5.0, 4.0], [5.2, 4.1],];

        let (leaders, labels) = leader_clustering(data.view(), 1.0, euclidean_distance).unwrap();

        // Should create 2 clusters
        assert_eq!(leaders.nrows(), 2);
        assert_eq!(labels.len(), 4);

        // Points 0,1 should be in one cluster, points 2,3 in another
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn test_leader_clustering_single_cluster() {
        let data = array![[1.0, 2.0], [1.2, 1.8], [1.1, 2.1], [0.9, 1.9],];

        let (leaders, labels) = leader_clustering(data.view(), 2.0, euclidean_distance).unwrap();

        // Should create 1 cluster with large threshold
        assert_eq!(leaders.nrows(), 1);
        assert!(labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn test_leader_class() {
        let data = array![[1.0, 2.0], [1.2, 1.8], [5.0, 4.0], [5.2, 4.1],];

        let mut leader = LeaderClustering::new(1.0).unwrap();
        let labels = leader.fit_predict(data.view()).unwrap();

        assert_eq!(leader.n_clusters(), 2);
        assert_eq!(labels.len(), 4);

        // Test prediction on new data
        let new_data = array![[1.1, 1.9], [5.1, 4.05]];
        let new_labels = leader.predict(new_data.view()).unwrap();
        assert_eq!(new_labels[0], labels[0]); // Close to first cluster
        assert_eq!(new_labels[1], labels[2]); // Close to second cluster
    }

    #[test]
    fn test_hierarchical_leader_tree() {
        let data = array![
            [1.0, 2.0],
            [1.2, 1.8],
            [5.0, 4.0],
            [5.2, 4.1],
            [10.0, 10.0],
            [10.2, 9.8],
        ];

        let thresholds = vec![6.0, 1.0];
        let tree = LeaderTree::build_hierarchical(data.view(), &thresholds).unwrap();

        // At threshold 6.0, should have 2 clusters (1,2 and 3,4,5,6)
        assert!(tree.roots.len() <= 3);
        assert!(tree.node_count() > tree.roots.len()); // Should have child nodes
    }

    #[test]
    fn test_invalid_threshold() {
        let data = array![[1.0, 2.0]];

        let result = leader_clustering(data.view(), -1.0, euclidean_distance);
        assert!(result.is_err());

        let result = LeaderClustering::<f64>::new(-1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_data() {
        let data: Array2<f64> = Array2::zeros((0, 2));

        let result = leader_clustering(data.view(), 1.0, euclidean_distance);
        assert!(result.is_err());
    }
}
