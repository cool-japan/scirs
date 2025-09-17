//! Adaptive online clustering with concept drift detection
//!
//! This module provides implementations of online clustering algorithms that
//! automatically adapt to changing data distributions, create new clusters when
//! needed, merge similar clusters, and detect concept drift in streaming data.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Zip};
use num_traits::{Float, FromPrimitive};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};
use crate::vq::euclidean_distance;

/// Configuration for adaptive online clustering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveOnlineConfig {
    /// Initial learning rate
    pub initial_learning_rate: f64,
    /// Minimum learning rate
    pub min_learning_rate: f64,
    /// Learning rate decay factor
    pub learning_rate_decay: f64,
    /// Forgetting factor for older data
    pub forgetting_factor: f64,
    /// Threshold for creating new clusters
    pub cluster_creation_threshold: f64,
    /// Maximum number of clusters allowed
    pub max_clusters: usize,
    /// Minimum cluster size before merging
    pub min_cluster_size: usize,
    /// Distance threshold for cluster merging
    pub merge_threshold: f64,
    /// Window size for concept drift detection
    pub concept_drift_window: usize,
    /// Threshold for detecting concept drift
    pub drift_detection_threshold: f64,
}

impl Default for AdaptiveOnlineConfig {
    fn default() -> Self {
        Self {
            initial_learning_rate: 0.1,
            min_learning_rate: 0.001,
            learning_rate_decay: 0.999,
            forgetting_factor: 0.95,
            cluster_creation_threshold: 2.0,
            max_clusters: 50,
            min_cluster_size: 10,
            merge_threshold: 0.5,
            concept_drift_window: 1000,
            drift_detection_threshold: 0.3,
        }
    }
}

/// Adaptive online clustering with concept drift detection
///
/// This algorithm automatically adapts to changing data distributions,
/// creates new clusters when needed, merges similar clusters, and detects
/// concept drift in streaming data.
pub struct AdaptiveOnlineClustering<F: Float> {
    config: AdaptiveOnlineConfig,
    clusters: Vec<OnlineCluster<F>>,
    learning_rate: f64,
    samples_processed: usize,
    recent_distances: VecDeque<f64>,
    drift_detector: ConceptDriftDetector,
}

/// Represents an online cluster with adaptive properties
#[derive(Debug, Clone)]
struct OnlineCluster<F: Float> {
    /// Cluster centroid
    centroid: Array1<F>,
    /// Number of points assigned to this cluster
    weight: f64,
    /// Timestamp of last update
    last_update: usize,
    /// Variance estimate for this cluster
    variance: f64,
    /// Cluster age (for aging/forgetting)
    age: usize,
    /// Recent assignment history
    recent_assignments: VecDeque<usize>,
}

/// Simple concept drift detector
#[derive(Debug, Clone)]
struct ConceptDriftDetector {
    /// Recent prediction errors
    recent_errors: VecDeque<f64>,
    /// Baseline error rate
    baseline_error: f64,
    /// Window size for drift detection
    window_size: usize,
}

impl<F: Float + FromPrimitive + Debug> AdaptiveOnlineClustering<F> {
    /// Create a new adaptive online clustering instance
    pub fn new(config: AdaptiveOnlineConfig) -> Self {
        Self {
            config: config.clone(),
            clusters: Vec::new(),
            learning_rate: config.initial_learning_rate,
            samples_processed: 0,
            recent_distances: VecDeque::with_capacity(config.concept_drift_window),
            drift_detector: ConceptDriftDetector {
                recent_errors: VecDeque::with_capacity(config.concept_drift_window),
                baseline_error: 1.0,
                window_size: config.concept_drift_window,
            },
        }
    }

    /// Process a single data point online
    pub fn partial_fit(&mut self, point: ArrayView1<F>) -> Result<usize> {
        self.samples_processed += 1;

        // Find nearest cluster
        let (nearest_cluster_idx, nearest_distance) = self.find_nearest_cluster(point);

        let assigned_cluster = if let Some(cluster_idx) = nearest_cluster_idx {
            let distance_threshold = F::from(self.config.cluster_creation_threshold).unwrap();

            if nearest_distance <= distance_threshold {
                // Update existing cluster
                self.update_cluster(cluster_idx, point)?;
                cluster_idx
            } else if self.clusters.len() < self.config.max_clusters {
                // Create new cluster
                self.create_new_cluster(point)?
            } else {
                // Force assignment to nearest cluster and update threshold
                self.update_cluster(cluster_idx, point)?;
                cluster_idx
            }
        } else {
            // No clusters exist, create first one
            self.create_new_cluster(point)?
        };

        // Update learning rate
        self.learning_rate = (self.learning_rate * self.config.learning_rate_decay)
            .max(self.config.min_learning_rate);

        // Track distance for concept drift detection
        self.recent_distances
            .push_back(nearest_distance.to_f64().unwrap_or(0.0));
        if self.recent_distances.len() > self.config.concept_drift_window {
            self.recent_distances.pop_front();
        }

        // Detect concept drift
        if self.samples_processed % 100 == 0 {
            self.detect_concept_drift()?;
        }

        // Periodic maintenance
        if self.samples_processed % 1000 == 0 {
            self.merge_similar_clusters()?;
            self.remove_old_clusters()?;
        }

        Ok(assigned_cluster)
    }

    /// Find the nearest cluster to a point
    fn find_nearest_cluster(&self, point: ArrayView1<F>) -> (Option<usize>, F) {
        if self.clusters.is_empty() {
            return (None, F::infinity());
        }

        let mut min_distance = F::infinity();
        let mut nearest_idx = 0;

        for (i, cluster) in self.clusters.iter().enumerate() {
            let distance = euclidean_distance(point, cluster.centroid.view());
            if distance < min_distance {
                min_distance = distance;
                nearest_idx = i;
            }
        }

        (Some(nearest_idx), min_distance)
    }

    /// Update an existing cluster with a new point
    fn update_cluster(&mut self, clusteridx: usize, point: ArrayView1<F>) -> Result<()> {
        let cluster = &mut self.clusters[clusteridx];

        // Update weight with forgetting factor
        cluster.weight = cluster.weight * self.config.forgetting_factor + 1.0;

        // Update centroid using online mean
        let learning_rate = F::from(self.learning_rate / cluster.weight).unwrap();

        Zip::from(&mut cluster.centroid)
            .and(point)
            .for_each(|centroid_val, &point_val| {
                let diff = point_val - *centroid_val;
                *centroid_val = *centroid_val + learning_rate * diff;
            });

        // Update variance estimate
        let distance = euclidean_distance(point, cluster.centroid.view());
        let distance_squared = distance * distance;
        cluster.variance = cluster.variance * 0.9 + distance_squared.to_f64().unwrap_or(0.0) * 0.1;

        // Update metadata
        cluster.last_update = self.samples_processed;
        cluster.age += 1;
        cluster.recent_assignments.push_back(self.samples_processed);

        if cluster.recent_assignments.len() > 100 {
            cluster.recent_assignments.pop_front();
        }

        Ok(())
    }

    /// Create a new cluster
    fn create_new_cluster(&mut self, point: ArrayView1<F>) -> Result<usize> {
        let new_cluster = OnlineCluster {
            centroid: point.to_owned(),
            weight: 1.0,
            last_update: self.samples_processed,
            variance: 0.0,
            age: 0,
            recent_assignments: VecDeque::new(),
        };

        self.clusters.push(new_cluster);
        Ok(self.clusters.len() - 1)
    }

    /// Detect concept drift in the data stream
    fn detect_concept_drift(&mut self) -> Result<()> {
        if self.recent_distances.len() < self.config.concept_drift_window / 2 {
            return Ok(());
        }

        // Calculate recent mean distance
        let recent_mean: f64 =
            self.recent_distances.iter().sum::<f64>() / self.recent_distances.len() as f64;

        // Update drift detector
        self.drift_detector.recent_errors.push_back(recent_mean);
        if self.drift_detector.recent_errors.len() > self.drift_detector.window_size {
            self.drift_detector.recent_errors.pop_front();
        }

        // Calculate current error rate
        let current_error: f64 = self.drift_detector.recent_errors.iter().sum::<f64>()
            / self.drift_detector.recent_errors.len() as f64;

        // Detect drift if current error is significantly higher than baseline
        if current_error
            > self.drift_detector.baseline_error * (1.0 + self.config.drift_detection_threshold)
        {
            // Concept drift detected - adapt by increasing learning rate temporarily
            self.learning_rate = (self.learning_rate * 2.0).min(0.5);
            self.drift_detector.baseline_error = current_error;
        } else {
            // Update baseline gradually
            self.drift_detector.baseline_error =
                self.drift_detector.baseline_error * 0.99 + current_error * 0.01;
        }

        Ok(())
    }

    /// Merge clusters that are too similar
    fn merge_similar_clusters(&mut self) -> Result<()> {
        let mut to_merge = Vec::new();
        let merge_threshold = F::from(self.config.merge_threshold).unwrap();

        // Find pairs of clusters to merge
        for i in 0..self.clusters.len() {
            for j in (i + 1)..self.clusters.len() {
                let distance = euclidean_distance(
                    self.clusters[i].centroid.view(),
                    self.clusters[j].centroid.view(),
                );

                if distance <= merge_threshold {
                    to_merge.push((i, j));
                }
            }
        }

        // Merge clusters (process in reverse order to maintain indices)
        for (i, j) in to_merge.into_iter().rev() {
            self.merge_clusters(i, j)?;
        }

        Ok(())
    }

    /// Merge two clusters
    fn merge_clusters(&mut self, i: usize, j: usize) -> Result<()> {
        if i >= self.clusters.len() || j >= self.clusters.len() || i == j {
            return Ok(());
        }

        let (cluster_i, cluster_j) = if i < j {
            let (left, right) = self.clusters.split_at_mut(j);
            (&mut left[i], &mut right[0])
        } else {
            let (left, right) = self.clusters.split_at_mut(i);
            (&mut right[0], &mut left[j])
        };

        // Weighted merge of centroids
        let total_weight = cluster_i.weight + cluster_j.weight;
        let weight_i = F::from(cluster_i.weight / total_weight).unwrap();
        let weight_j = F::from(cluster_j.weight / total_weight).unwrap();

        Zip::from(&mut cluster_i.centroid)
            .and(&cluster_j.centroid)
            .for_each(|cent_i, &cent_j| {
                *cent_i = *cent_i * weight_i + cent_j * weight_j;
            });

        // Merge other properties
        cluster_i.weight = total_weight;
        cluster_i.variance = (cluster_i.variance + cluster_j.variance) / 2.0;
        cluster_i.age = cluster_i.age.max(cluster_j.age);
        cluster_i.last_update = cluster_i.last_update.max(cluster_j.last_update);

        // Remove the merged cluster
        let remove_idx = if i < j { j } else { i };
        self.clusters.remove(remove_idx);

        Ok(())
    }

    /// Remove old, inactive clusters
    fn remove_old_clusters(&mut self) -> Result<()> {
        let current_time = self.samples_processed;
        let max_age = 10000; // Maximum age before considering removal

        self.clusters.retain(|cluster| {
            let age_ok = cluster.age < max_age;
            let recent_activity = current_time - cluster.last_update < 5000;
            let sufficient_size = cluster.weight >= self.config.min_cluster_size as f64;

            age_ok && (recent_activity || sufficient_size)
        });

        Ok(())
    }

    /// Predict cluster assignment for new data
    pub fn predict(&self, point: ArrayView1<F>) -> Result<usize> {
        let (nearest_cluster_idx_, _distance) = self.find_nearest_cluster(point);

        nearest_cluster_idx_.ok_or_else(|| {
            ClusteringError::InvalidInput("No clusters available for prediction".to_string())
        })
    }

    /// Get current cluster centroids
    pub fn cluster_centers(&self) -> Array2<F> {
        if self.clusters.is_empty() {
            return Array2::zeros((0, 0));
        }

        let n_features = self.clusters[0].centroid.len();
        let mut centers = Array2::zeros((self.clusters.len(), n_features));

        for (i, cluster) in self.clusters.iter().enumerate() {
            centers.row_mut(i).assign(&cluster.centroid);
        }

        centers
    }

    /// Get cluster information for analysis
    pub fn cluster_info(&self) -> Vec<(f64, f64, usize)> {
        self.clusters
            .iter()
            .map(|cluster| (cluster.weight, cluster.variance, cluster.age))
            .collect()
    }

    /// Get number of active clusters
    pub fn n_clusters(&self) -> usize {
        self.clusters.len()
    }
}

/// Convenience function for adaptive online clustering
pub fn adaptive_online_clustering<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    config: Option<AdaptiveOnlineConfig>,
) -> Result<(Array2<F>, Array1<usize>)> {
    let config = config.unwrap_or_default();
    let mut clusterer = AdaptiveOnlineClustering::new(config);

    let n_samples = data.nrows();
    let mut labels = Array1::zeros(n_samples);

    // Process data points sequentially
    for (i, point) in data.rows().into_iter().enumerate() {
        labels[i] = clusterer.partial_fit(point)?;
    }

    let centers = clusterer.cluster_centers();
    Ok((centers, labels))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_adaptive_online_config_default() {
        let config = AdaptiveOnlineConfig::default();
        assert_eq!(config.initial_learning_rate, 0.1);
        assert_eq!(config.max_clusters, 50);
        assert_eq!(config.concept_drift_window, 1000);
    }

    #[test]
    fn test_adaptive_online_clustering_creation() {
        let config = AdaptiveOnlineConfig::default();
        let clusterer = AdaptiveOnlineClustering::<f64>::new(config);
        assert_eq!(clusterer.n_clusters(), 0);
    }

    #[test]
    fn test_adaptive_online_clustering_simple() {
        let data = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 1.0, 10.0, 10.0, 11.0, 11.0])
            .unwrap();

        let config = AdaptiveOnlineConfig {
            cluster_creation_threshold: 2.0,
            max_clusters: 10,
            ..Default::default()
        };

        let result = adaptive_online_clustering(data.view(), Some(config));
        assert!(result.is_ok());

        let (centers, labels) = result.unwrap();
        assert_eq!(labels.len(), 4);
        assert!(centers.nrows() <= 4); // Should create clusters as needed
    }

    #[test]
    fn test_online_cluster_creation() {
        let config = AdaptiveOnlineConfig::default();
        let mut clusterer = AdaptiveOnlineClustering::<f64>::new(config);

        let point = Array1::from_vec(vec![1.0, 2.0]);
        let cluster_id = clusterer.partial_fit(point.view()).unwrap();

        assert_eq!(cluster_id, 0);
        assert_eq!(clusterer.n_clusters(), 1);
    }

    #[test]
    fn test_concept_drift_detection() {
        let config = AdaptiveOnlineConfig {
            concept_drift_window: 10,
            drift_detection_threshold: 0.1,
            ..Default::default()
        };

        let mut clusterer = AdaptiveOnlineClustering::<f64>::new(config);

        // Process some initial points
        for i in 0..5 {
            let point = Array1::from_vec(vec![i as f64, i as f64]);
            clusterer.partial_fit(point.view()).unwrap();
        }

        // The drift detection should run without errors
        assert!(clusterer.detect_concept_drift().is_ok());
    }

    #[test]
    fn test_cluster_merging() {
        let config = AdaptiveOnlineConfig {
            merge_threshold: 1.0,
            cluster_creation_threshold: 0.5,
            ..Default::default()
        };

        let mut clusterer = AdaptiveOnlineClustering::<f64>::new(config);

        // Create two close clusters
        let point1 = Array1::from_vec(vec![0.0, 0.0]);
        let point2 = Array1::from_vec(vec![0.3, 0.3]);

        clusterer.partial_fit(point1.view()).unwrap();
        clusterer.partial_fit(point2.view()).unwrap();

        // Initial clusters should exist
        let initial_clusters = clusterer.n_clusters();

        // Force merge check
        clusterer.merge_similar_clusters().unwrap();

        // Clusters might be merged if they're close enough
        assert!(clusterer.n_clusters() <= initial_clusters);
    }
}
