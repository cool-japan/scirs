//! Clustering model implementations for serialization
//!
//! This module contains serializable model structures for different
//! clustering algorithms including K-means, DBSCAN, hierarchical clustering, etc.

use crate::error::{ClusteringError, Result};
use crate::leader::{LeaderNode, LeaderTree};
use ndarray::{Array1, Array2, ArrayView2};
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::core::SerializableModel;

/// K-means clustering model that can be serialized
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct KMeansModel {
    /// Cluster centroids
    pub centroids: Array2<f64>,
    /// Number of clusters
    pub n_clusters: usize,
    /// Number of iterations performed
    pub n_iter: usize,
    /// Sum of squared distances
    pub inertia: f64,
    /// Cluster labels for training data (optional)
    pub labels: Option<Array1<usize>>,
}

impl SerializableModel for KMeansModel {}

impl KMeansModel {
    /// Create a new K-means model
    pub fn new(
        centroids: Array2<f64>,
        n_clusters: usize,
        n_iter: usize,
        inertia: f64,
        labels: Option<Array1<usize>>,
    ) -> Self {
        Self {
            centroids,
            n_clusters,
            n_iter,
            inertia,
            labels,
        }
    }

    /// Predict cluster labels for new data
    pub fn predict(&self, data: ArrayView2<f64>) -> Result<Array1<usize>> {
        let n_samples = data.nrows();
        let mut labels = Array1::zeros(n_samples);

        for (i, sample) in data.rows().into_iter().enumerate() {
            let mut min_distance = f64::INFINITY;
            let mut closest_cluster = 0;

            for (j, centroid) in self.centroids.rows().into_iter().enumerate() {
                let distance = sample
                    .iter()
                    .zip(centroid.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();

                if distance < min_distance {
                    min_distance = distance;
                    closest_cluster = j;
                }
            }

            labels[i] = closest_cluster;
        }

        Ok(labels)
    }

    /// Get the closest cluster center for a single point
    pub fn predict_single(&self, point: &[f64]) -> Result<usize> {
        if point.len() != self.centroids.ncols() {
            return Err(ClusteringError::InvalidInput(
                "Point dimensions must match centroid dimensions".to_string(),
            ));
        }

        let mut min_distance = f64::INFINITY;
        let mut closest_cluster = 0;

        for (j, centroid) in self.centroids.rows().into_iter().enumerate() {
            let distance = point
                .iter()
                .zip(centroid.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            if distance < min_distance {
                min_distance = distance;
                closest_cluster = j;
            }
        }

        Ok(closest_cluster)
    }
}

/// Hierarchical clustering result that can be serialized
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HierarchicalModel {
    /// Linkage matrix
    pub linkage: Array2<f64>,
    /// Number of original observations
    pub n_observations: usize,
    /// Method used for linkage
    pub method: String,
    /// Dendrogram labels (optional)
    pub labels: Option<Vec<String>>,
}

impl SerializableModel for HierarchicalModel {}

impl HierarchicalModel {
    /// Create a new hierarchical clustering model
    pub fn new(
        linkage: Array2<f64>,
        n_observations: usize,
        method: String,
        labels: Option<Vec<String>>,
    ) -> Self {
        Self {
            linkage,
            n_observations,
            method,
            labels,
        }
    }

    /// Export dendrogram to Newick format
    pub fn to_newick(&self) -> Result<String> {
        let mut newick = String::new();
        let nnodes = self.linkage.nrows();

        if nnodes == 0 {
            return Ok("();".to_string());
        }

        self.validate_linkage_matrix()?;
        self.build_newick_recursive(nnodes + self.n_observations - 1, &mut newick)?;

        newick.push_str(";");
        Ok(newick)
    }

    /// Validate linkage matrix for consistency
    fn validate_linkage_matrix(&self) -> Result<()> {
        let nnodes = self.linkage.nrows();

        for i in 0..nnodes {
            let left = self.linkage[[i, 0]] as usize;
            let right = self.linkage[[i, 1]] as usize;
            let distance = self.linkage[[i, 2]];

            if left >= self.n_observations + i || right >= self.n_observations + i {
                return Err(ClusteringError::InvalidInput(format!(
                    "Invalid node indices in linkage matrix at row {}: left={}, right={}",
                    i, left, right
                )));
            }

            if distance < 0.0 {
                return Err(ClusteringError::InvalidInput(format!(
                    "Negative distance in linkage matrix at row {}: {}",
                    i, distance
                )));
            }
        }

        Ok(())
    }

    /// Build Newick string recursively
    fn build_newick_recursive(&self, nodeidx: usize, newick: &mut String) -> Result<()> {
        if nodeidx < self.n_observations {
            if let Some(ref labels) = self.labels {
                newick.push_str(&labels[nodeidx]);
            } else {
                newick.push_str(&nodeidx.to_string());
            }
        } else {
            let row_idx = nodeidx - self.n_observations;
            if row_idx >= self.linkage.nrows() {
                return Err(ClusteringError::InvalidInput(
                    "Invalid node index".to_string(),
                ));
            }

            let left = self.linkage[[row_idx, 0]] as usize;
            let right = self.linkage[[row_idx, 1]] as usize;
            let distance = self.linkage[[row_idx, 2]];

            newick.push('(');
            self.build_newick_recursive(left, newick)?;
            newick.push(':');
            newick.push_str(&format!("{:.6}", distance / 2.0));
            newick.push(',');
            self.build_newick_recursive(right, newick)?;
            newick.push(':');
            newick.push_str(&format!("{:.6}", distance / 2.0));
            newick.push(')');
        }

        Ok(())
    }

    /// Export dendrogram to JSON format
    pub fn to_json_tree(&self) -> Result<serde_json::Value> {
        use serde_json::json;

        let nnodes = self.linkage.nrows();
        if nnodes == 0 {
            return Ok(json!({}));
        }

        self.build_json_recursive(nnodes + self.n_observations - 1)
    }

    fn build_json_recursive(&self, nodeidx: usize) -> Result<serde_json::Value> {
        use serde_json::json;

        if nodeidx < self.n_observations {
            let name = if let Some(ref labels) = self.labels {
                labels[nodeidx].clone()
            } else {
                nodeidx.to_string()
            };

            Ok(json!({
                "name": name,
                "type": "leaf",
                "index": nodeidx
            }))
        } else {
            let row_idx = nodeidx - self.n_observations;
            if row_idx >= self.linkage.nrows() {
                return Err(ClusteringError::InvalidInput(
                    "Invalid node index".to_string(),
                ));
            }

            let left = self.linkage[[row_idx, 0]] as usize;
            let right = self.linkage[[row_idx, 1]] as usize;
            let distance = self.linkage[[row_idx, 2]];

            let left_child = self.build_json_recursive(left)?;
            let right_child = self.build_json_recursive(right)?;

            Ok(json!({
                "type": "internal",
                "distance": distance,
                "children": [left_child, right_child]
            }))
        }
    }
}

/// DBSCAN model that can be serialized
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DBSCANModel {
    /// Core sample indices
    pub core_sample_indices: Array1<usize>,
    /// Cluster labels
    pub labels: Array1<i32>,
    /// Epsilon parameter
    pub eps: f64,
    /// Min samples parameter
    pub min_samples: usize,
}

impl SerializableModel for DBSCANModel {}

impl DBSCANModel {
    /// Create a new DBSCAN model
    pub fn new(
        core_sample_indices: Array1<usize>,
        labels: Array1<i32>,
        eps: f64,
        min_samples: usize,
    ) -> Self {
        Self {
            core_sample_indices,
            labels,
            eps,
            min_samples,
        }
    }

    /// Get number of clusters (excluding noise)
    pub fn n_clusters(&self) -> usize {
        self.labels.iter().filter(|&&label| label >= 0).count()
    }

    /// Get noise point indices
    pub fn noise_indices(&self) -> Vec<usize> {
        self.labels
            .iter()
            .enumerate()
            .filter_map(|(i, &label)| if label == -1 { Some(i) } else { None })
            .collect()
    }
}

/// Mean Shift model that can be serialized
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MeanShiftModel {
    /// Cluster centers
    pub cluster_centers: Array2<f64>,
    /// Bandwidth parameter
    pub bandwidth: f64,
    /// Cluster labels (optional)
    pub labels: Option<Array1<usize>>,
}

impl SerializableModel for MeanShiftModel {}

/// Spectral clustering model that can be serialized
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SpectralModel {
    /// Eigenvectors used for clustering
    pub eigenvectors: Array2<f64>,
    /// Eigenvalues
    pub eigenvalues: Array1<f64>,
    /// Cluster labels
    pub labels: Array1<usize>,
    /// Number of clusters
    pub n_clusters: usize,
    /// Affinity matrix parameters
    pub affinity: String,
    /// Gamma parameter for RBF kernel
    pub gamma: Option<f64>,
}

impl SerializableModel for SpectralModel {}

impl SpectralModel {
    /// Create a new spectral clustering model
    pub fn new(
        eigenvectors: Array2<f64>,
        eigenvalues: Array1<f64>,
        labels: Array1<usize>,
        n_clusters: usize,
        affinity: String,
        gamma: Option<f64>,
    ) -> Self {
        Self {
            eigenvectors,
            eigenvalues,
            labels,
            n_clusters,
            affinity,
            gamma,
        }
    }

    /// Predict cluster labels for new data
    pub fn predict(&self, data: ArrayView2<f64>) -> Result<Array1<usize>> {
        // Simple prediction based on closest eigenvector projection
        let n_samples = data.nrows();
        let mut labels = Array1::zeros(n_samples);

        for (i, sample) in data.rows().into_iter().enumerate() {
            let mut best_distance = f64::INFINITY;
            let mut best_cluster = 0;

            for cluster_id in 0..self.n_clusters {
                // Simple distance to cluster center in eigenspace
                let distance = sample
                    .iter()
                    .zip(
                        self.eigenvectors
                            .row(cluster_id % self.eigenvectors.nrows())
                            .iter(),
                    )
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();

                if distance < best_distance {
                    best_distance = distance;
                    best_cluster = cluster_id;
                }
            }

            labels[i] = best_cluster;
        }

        Ok(labels)
    }
}

/// Generic clustering model trait
pub trait ClusteringModel: SerializableModel {
    /// Get the number of clusters
    fn n_clusters(&self) -> usize;

    /// Predict cluster labels for new data
    fn predict(&self, data: ArrayView2<f64>) -> Result<Array1<usize>>;

    /// Get model summary as JSON
    fn summary(&self) -> Result<serde_json::Value>;
}

impl ClusteringModel for KMeansModel {
    fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    fn predict(&self, data: ArrayView2<f64>) -> Result<Array1<usize>> {
        self.predict(data)
    }

    fn summary(&self) -> Result<serde_json::Value> {
        Ok(serde_json::json!({
            "algorithm": "K-Means",
            "n_clusters": self.n_clusters,
            "n_features": self.centroids.ncols(),
            "n_iterations": self.n_iter,
            "inertia": self.inertia,
            "has_training_labels": self.labels.is_some()
        }))
    }
}

impl ClusteringModel for DBSCANModel {
    fn n_clusters(&self) -> usize {
        self.labels
            .iter()
            .filter(|&&x| x >= 0)
            .map(|&x| x as usize)
            .max()
            .map(|x| x + 1)
            .unwrap_or(0)
    }

    fn predict(&self, data: ArrayView2<f64>) -> Result<Array1<usize>> {
        self.predict(data)
    }

    fn summary(&self) -> Result<serde_json::Value> {
        let n_clusters = self.n_clusters();
        let n_noise = self.labels.iter().filter(|&&x| x == -1).count();

        Ok(serde_json::json!({
            "algorithm": "DBSCAN",
            "n_clusters": n_clusters,
            "n_core_samples": self.core_sample_indices.len(),
            "n_noise_points": n_noise,
            "eps": self.eps,
            "min_samples": self.min_samples
        }))
    }
}

impl ClusteringModel for SpectralModel {
    fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    fn predict(&self, data: ArrayView2<f64>) -> Result<Array1<usize>> {
        self.predict(data)
    }

    fn summary(&self) -> Result<serde_json::Value> {
        Ok(serde_json::json!({
            "algorithm": "Spectral Clustering",
            "n_clusters": self.n_clusters,
            "n_eigenvectors": self.eigenvectors.ncols(),
            "affinity": self.affinity,
            "gamma": self.gamma
        }))
    }
}

impl MeanShiftModel {
    /// Create a new Mean Shift model
    pub fn new(
        cluster_centers: Array2<f64>,
        bandwidth: f64,
        labels: Option<Array1<usize>>,
    ) -> Self {
        Self {
            cluster_centers,
            bandwidth,
            labels,
        }
    }

    /// Get number of clusters
    pub fn n_clusters(&self) -> usize {
        self.cluster_centers.nrows()
    }
}

/// Leader clustering model that can be serialized
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LeaderModel {
    /// Leader nodes
    pub leaders: Vec<LeaderNode<f64>>,
    /// Threshold parameter
    pub threshold: f64,
    /// Distance metric used
    pub metric: String,
}

impl SerializableModel for LeaderModel {}

impl LeaderModel {
    /// Create a new Leader model
    pub fn new(leaders: Vec<LeaderNode<f64>>, threshold: f64, metric: String) -> Self {
        Self {
            leaders,
            threshold,
            metric,
        }
    }

    /// Get number of clusters
    pub fn n_clusters(&self) -> usize {
        self.leaders.len()
    }

    /// Predict cluster for a new point
    pub fn predict_single(&self, point: &[f64]) -> Result<Option<usize>> {
        let mut best_leader = None;
        let mut min_distance = self.threshold;

        for (i, leader) in self.leaders.iter().enumerate() {
            let distance = match self.metric.as_str() {
                "euclidean" => point
                    .iter()
                    .zip(leader.leader.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt(),
                "manhattan" => point
                    .iter()
                    .zip(leader.leader.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f64>(),
                _ => return Err(ClusteringError::InvalidInput("Unknown metric".to_string())),
            };

            if distance < min_distance {
                min_distance = distance;
                best_leader = Some(i);
            }
        }

        Ok(best_leader)
    }
}

/// Leader Tree clustering model that can be serialized
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LeaderTreeModel<F: Float> {
    /// Root of the leader tree
    pub tree: LeaderTree<F>,
    /// Threshold parameter
    pub threshold: F,
    /// Distance metric used
    pub metric: String,
}

impl<F: Float + Serialize + for<'de> Deserialize<'de>> SerializableModel for LeaderTreeModel<F> {}

/// Affinity Propagation model that can be serialized
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AffinityPropagationModel {
    /// Cluster centers (exemplars)
    pub cluster_centers: Array2<f64>,
    /// Cluster labels
    pub labels: Array1<i32>,
    /// Affinity matrix
    pub affinity_matrix: Array2<f64>,
    /// Converged flag
    pub converged: bool,
    /// Number of iterations
    pub n_iter: usize,
}

impl SerializableModel for AffinityPropagationModel {}

impl AffinityPropagationModel {
    /// Create a new Affinity Propagation model
    pub fn new(
        cluster_centers: Array2<f64>,
        labels: Array1<i32>,
        affinity_matrix: Array2<f64>,
        converged: bool,
        n_iter: usize,
    ) -> Self {
        Self {
            cluster_centers,
            labels,
            affinity_matrix,
            converged,
            n_iter,
        }
    }

    /// Get number of clusters
    pub fn n_clusters(&self) -> usize {
        self.cluster_centers.nrows()
    }
}

/// BIRCH clustering model that can be serialized
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BirchModel {
    /// Cluster centroids
    pub centroids: Array2<f64>,
    /// Threshold parameter
    pub threshold: f64,
    /// Branching factor
    pub branching_factor: usize,
    /// Number of subclusters
    pub n_subclusters: usize,
}

impl SerializableModel for BirchModel {}

impl BirchModel {
    /// Create a new BIRCH model
    pub fn new(
        centroids: Array2<f64>,
        threshold: f64,
        branching_factor: usize,
        n_subclusters: usize,
    ) -> Self {
        Self {
            centroids,
            threshold,
            branching_factor,
            n_subclusters,
        }
    }

    /// Get number of clusters
    pub fn n_clusters(&self) -> usize {
        self.centroids.nrows()
    }
}

/// Gaussian Mixture Model that can be serialized
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GMMModel {
    /// Mixture weights
    pub weights: Array1<f64>,
    /// Component means
    pub means: Array2<f64>,
    /// Component covariances
    pub covariances: Vec<Array2<f64>>,
    /// Number of components
    pub n_components: usize,
    /// Covariance type
    pub covariance_type: String,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// Converged flag
    pub converged: bool,
    /// Number of iterations
    pub n_iter: usize,
}

impl SerializableModel for GMMModel {}

impl GMMModel {
    /// Create a new GMM model
    pub fn new(
        weights: Array1<f64>,
        means: Array2<f64>,
        covariances: Vec<Array2<f64>>,
        n_components: usize,
        covariance_type: String,
        log_likelihood: f64,
        converged: bool,
        n_iter: usize,
    ) -> Self {
        Self {
            weights,
            means,
            covariances,
            n_components,
            covariance_type,
            log_likelihood,
            converged,
            n_iter,
        }
    }

    /// Predict cluster probabilities for new data
    pub fn predict_proba(&self, data: ArrayView2<f64>) -> Result<Array2<f64>> {
        let n_samples = data.nrows();
        let mut probabilities = Array2::zeros((n_samples, self.n_components));

        for (i, sample) in data.rows().into_iter().enumerate() {
            for j in 0..self.n_components {
                let mean = self.means.row(j);
                let diff: Vec<f64> = sample.iter().zip(mean.iter()).map(|(a, b)| a - b).collect();

                // Simplified probability calculation (would need proper multivariate normal)
                let distance = diff.iter().map(|x| x * x).sum::<f64>().sqrt();
                probabilities[[i, j]] = self.weights[j] * (-distance / 2.0).exp();
            }
        }

        // Normalize probabilities
        for i in 0..n_samples {
            let sum: f64 = probabilities.row(i).sum();
            if sum > 0.0 {
                for j in 0..self.n_components {
                    probabilities[[i, j]] /= sum;
                }
            }
        }

        Ok(probabilities)
    }
}

/// Spectral clustering model that can be serialized
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SpectralClusteringModel {
    /// Cluster labels
    pub labels: Array1<usize>,
    /// Affinity matrix
    pub affinity_matrix: Array2<f64>,
    /// Eigenvalues
    pub eigenvalues: Array1<f64>,
    /// Eigenvectors
    pub eigenvectors: Array2<f64>,
    /// Number of clusters
    pub n_clusters: usize,
}

impl SerializableModel for SpectralClusteringModel {}

impl SpectralClusteringModel {
    /// Create a new Spectral clustering model
    pub fn new(
        labels: Array1<usize>,
        affinity_matrix: Array2<f64>,
        eigenvalues: Array1<f64>,
        eigenvectors: Array2<f64>,
        n_clusters: usize,
    ) -> Self {
        Self {
            labels,
            affinity_matrix,
            eigenvalues,
            eigenvectors,
            n_clusters,
        }
    }
}

// Conversion functions for creating models from algorithm results

/// Convert K-means results to serializable model
pub fn kmeans_to_model(
    centroids: Array2<f64>,
    labels: Option<Array1<usize>>,
    n_iter: usize,
    inertia: f64,
) -> KMeansModel {
    let n_clusters = centroids.nrows();
    KMeansModel::new(centroids, n_clusters, n_iter, inertia, labels)
}

/// Convert DBSCAN results to serializable model
pub fn dbscan_to_model(
    core_sample_indices: Vec<usize>,
    components: Array2<f64>,
    labels: Array1<i32>,
    eps: f64,
    min_samples: usize,
) -> DBSCANModel {
    DBSCANModel::new(
        Array1::from_vec(core_sample_indices),
        labels,
        eps,
        min_samples,
    )
}

/// Convert hierarchical clustering results to serializable model
pub fn hierarchy_to_model(
    n_clusters: usize,
    labels: Array1<usize>,
    linkage_matrix: Array2<f64>,
    distances: Vec<f64>,
) -> HierarchicalModel {
    HierarchicalModel::new(linkage_matrix, n_clusters, "ward".to_string(), None)
}

/// Convert GMM results to serializable model
pub fn gmm_to_model(
    weights: Array1<f64>,
    means: Array2<f64>,
    covariances: Vec<Array2<f64>>,
    n_components: usize,
    covariance_type: String,
    log_likelihood: f64,
    converged: bool,
    n_iter: usize,
) -> GMMModel {
    GMMModel::new(
        weights,
        means,
        covariances,
        n_components,
        covariance_type,
        log_likelihood,
        converged,
        n_iter,
    )
}

/// Convert Mean Shift results to serializable model
pub fn meanshift_to_model(
    cluster_centers: Array2<f64>,
    labels: Array1<usize>,
    bandwidth: f64,
    n_iter: usize,
) -> MeanShiftModel {
    MeanShiftModel::new(cluster_centers, bandwidth, Some(labels))
}

/// Convert Affinity Propagation results to serializable model
pub fn affinity_propagation_to_model(
    exemplars: Vec<usize>,
    labels: Array1<i32>,
    damping: f64,
    preference: f64,
    n_iter: usize,
) -> AffinityPropagationModel {
    // Extract cluster centers from exemplars
    let n_clusters = exemplars.len();
    let n_features = if n_clusters > 0 { 2 } else { 0 }; // Default assumption
    let cluster_centers = Array2::zeros((n_clusters, n_features));
    let affinity_matrix = Array2::zeros((labels.len(), labels.len()));

    AffinityPropagationModel::new(cluster_centers, labels, affinity_matrix, true, n_iter)
}

/// Convert BIRCH results to serializable model
pub fn birch_to_model(
    centroids: Array2<f64>,
    threshold: f64,
    branching_factor: usize,
    n_subclusters: usize,
) -> BirchModel {
    BirchModel::new(centroids, threshold, branching_factor, n_subclusters)
}

/// Convert Leader clustering results to serializable model
pub fn leader_to_model(
    leaders: Vec<LeaderNode<f64>>,
    threshold: f64,
    distance_metric: String,
) -> LeaderModel {
    // Convert LeaderNode to LeaderNode<f64> if needed, or use directly

    LeaderModel {
        leaders,
        threshold,
        metric: distance_metric,
    }
}

/// Convert Leader Tree results to serializable model
pub fn leadertree_to_model(
    tree: Option<LeaderTree<f64>>,
    threshold: f64,
    max_depth: usize,
) -> LeaderTreeModel<f64> {
    LeaderTreeModel {
        tree: tree.unwrap_or_else(|| LeaderTree {
            roots: Vec::new(),
            threshold,
        }),
        threshold,
        metric: "euclidean".to_string(),
    }
}

/// Convert Spectral clustering results to serializable model
pub fn spectral_clustering_to_model(
    labels: Array1<usize>,
    affinity_matrix: Array2<f64>,
    eigenvalues: Array1<f64>,
    eigenvectors: Array2<f64>,
    n_clusters: usize,
) -> SpectralClusteringModel {
    SpectralClusteringModel::new(
        labels,
        affinity_matrix,
        eigenvalues,
        eigenvectors,
        n_clusters,
    )
}

// Save functions for convenience

/// Save K-means model to file
pub fn save_kmeans<P: AsRef<std::path::Path>>(model: &KMeansModel, path: P) -> Result<()> {
    model.save_to_file(path)
}

/// Save DBSCAN model to file
pub fn save_dbscan<P: AsRef<std::path::Path>>(model: &DBSCANModel, path: P) -> Result<()> {
    model.save_to_file(path)
}

/// Save hierarchical clustering model to file
pub fn save_hierarchy<P: AsRef<std::path::Path>>(model: &HierarchicalModel, path: P) -> Result<()> {
    model.save_to_file(path)
}

/// Save GMM model to file
pub fn save_gmm<P: AsRef<std::path::Path>>(model: &GMMModel, path: P) -> Result<()> {
    model.save_to_file(path)
}

/// Save Mean Shift model to file
pub fn save_meanshift<P: AsRef<std::path::Path>>(model: &MeanShiftModel, path: P) -> Result<()> {
    model.save_to_file(path)
}

/// Save Affinity Propagation model to file
pub fn save_affinity_propagation<P: AsRef<std::path::Path>>(
    exemplars: Vec<usize>,
    labels: Array1<i32>,
    damping: f64,
    preference: f64,
    n_iter: usize,
    path: P,
) -> Result<()> {
    let model = affinity_propagation_to_model(exemplars, labels, damping, preference, n_iter);
    model.save_to_file(path)
}

/// Save BIRCH model to file
pub fn save_birch<P: AsRef<std::path::Path>>(model: &BirchModel, path: P) -> Result<()> {
    model.save_to_file(path)
}

/// Save Leader clustering model to file
pub fn save_leader<P: AsRef<std::path::Path>>(model: &LeaderModel, path: P) -> Result<()> {
    model.save_to_file(path)
}

/// Save Leader Tree model to file
pub fn save_leadertree<
    F: Float + Serialize + for<'de> serde::Deserialize<'de>,
    P: AsRef<std::path::Path>,
>(
    model: &LeaderTreeModel<F>,
    path: P,
) -> Result<()> {
    model.save_to_file(path)
}

/// Save Spectral clustering model to file
pub fn save_spectral_clustering<P: AsRef<std::path::Path>>(
    model: &SpectralClusteringModel,
    path: P,
) -> Result<()> {
    model.save_to_file(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_kmeans_model_predict() {
        let centroids = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let model = KMeansModel::new(centroids, 2, 10, 0.5, None);

        let data = Array2::from_shape_vec((2, 2), vec![0.1, 0.1, 0.9, 0.9]).unwrap();
        let labels = model.predict(data.view()).unwrap();

        assert_eq!(labels[0], 0); // Closer to first centroid
        assert_eq!(labels[1], 1); // Closer to second centroid
    }

    #[test]
    fn test_dbscan_model_clusters() {
        let core_indices = Array1::from_vec(vec![0, 1, 2]);
        let labels = Array1::from_vec(vec![0, 0, 1, -1]);
        let model = DBSCANModel::new(core_indices, labels, 0.5, 2);

        assert_eq!(model.n_clusters(), 3); // Points with labels 0, 0, 1 (excluding -1)
        assert_eq!(model.noise_indices(), vec![3]); // Point with label -1
    }

    #[test]
    fn test_hierarchical_model_newick() {
        let linkage = Array2::from_shape_vec((1, 3), vec![0.0, 1.0, 0.5]).unwrap();
        let model = HierarchicalModel::new(linkage, 2, "ward".to_string(), None);

        let newick = model.to_newick().unwrap();
        assert!(newick.contains("("));
        assert!(newick.contains(")"));
        assert!(newick.ends_with(";"));
    }
}
