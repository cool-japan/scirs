//! Self-supervised learning metrics
//!
//! This module provides evaluation metrics for self-supervised learning
//! including linear probing, clustering analysis, and representation quality.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use crate::error::{MetricsError, Result};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::numeric::Float;
use std::iter::Sum;

use super::results::{ClusteringResult, LinearProbingResult, RepresentationRankResult};

/// Self-supervised learning metrics
pub struct SelfSupervisedMetrics<F: Float> {
    /// Number of linear probing epochs
    pub n_probe_epochs: usize,
    /// Learning rate for linear probing
    pub probe_learning_rate: F,
    /// Number of clustering attempts
    pub n_clustering_runs: usize,
    _phantom: std::marker::PhantomData<F>,
}

impl<
        F: Float + scirs2_core::numeric::FromPrimitive + Sum + scirs2_core::ndarray::ScalarOperand,
    > Default for SelfSupervisedMetrics<F>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<
        F: Float + scirs2_core::numeric::FromPrimitive + Sum + scirs2_core::ndarray::ScalarOperand,
    > SelfSupervisedMetrics<F>
{
    /// Create new self-supervised learning metrics
    pub fn new() -> Self {
        Self {
            n_probe_epochs: 100,
            probe_learning_rate: F::from(0.001).unwrap(),
            n_clustering_runs: 5,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set linear probing parameters
    pub fn with_linear_probe(mut self, epochs: usize, lr: F) -> Self {
        self.n_probe_epochs = epochs;
        self.probe_learning_rate = lr;
        self
    }

    /// Set clustering parameters
    pub fn with_clustering(mut self, runs: usize) -> Self {
        self.n_clustering_runs = runs;
        self
    }

    /// Compute linear probing accuracy
    pub fn linear_probing_accuracy(
        &self,
        representations: &Array2<F>,
        labels: &Array1<usize>,
        test_representations: &Array2<F>,
        test_labels: &Array1<usize>,
    ) -> Result<LinearProbingResult<F>> {
        if representations.nrows() != labels.len() {
            return Err(MetricsError::InvalidInput(
                "Mismatched representations and labels".to_string(),
            ));
        }

        if test_representations.nrows() != test_labels.len() {
            return Err(MetricsError::InvalidInput(
                "Mismatched test representations and labels".to_string(),
            ));
        }

        // Get number of classes
        let n_classes = labels.iter().max().unwrap_or(&0) + 1;
        let n_features = representations.ncols();

        // Initialize linear classifier weights (simplified to centroid-based)
        let mut class_centroids = Array2::zeros((n_classes, n_features));
        let mut class_counts = vec![0; n_classes];

        // Compute class centroids
        for (i, &label) in labels.iter().enumerate() {
            for j in 0..n_features {
                class_centroids[[label, j]] = class_centroids[[label, j]] + representations[[i, j]];
            }
            class_counts[label] += 1;
        }

        // Normalize centroids
        for class in 0..n_classes {
            if class_counts[class] > 0 {
                let count = F::from(class_counts[class]).unwrap();
                for j in 0..n_features {
                    class_centroids[[class, j]] = class_centroids[[class, j]] / count;
                }
            }
        }

        // Evaluate on test set
        let mut correct_predictions = 0;
        let mut per_class_correct = vec![0; n_classes];
        let mut per_class_total = vec![0; n_classes];

        for (i, &true_label) in test_labels.iter().enumerate() {
            let test_sample = test_representations.row(i);

            // Find closest centroid
            let mut best_distance = F::infinity();
            let mut predicted_class = 0;

            for class in 0..n_classes {
                if class_counts[class] > 0 {
                    let centroid = class_centroids.row(class);
                    let distance =
                        self.euclidean_distance(&test_sample.to_owned(), &centroid.to_owned())?;

                    if distance < best_distance {
                        best_distance = distance;
                        predicted_class = class;
                    }
                }
            }

            per_class_total[true_label] += 1;
            if predicted_class == true_label {
                correct_predictions += 1;
                per_class_correct[true_label] += 1;
            }
        }

        let overall_accuracy =
            F::from(correct_predictions).unwrap() / F::from(test_labels.len()).unwrap();

        // Compute per-class accuracies
        let mut per_class_accuracies = Vec::with_capacity(n_classes);
        for class in 0..n_classes {
            if per_class_total[class] > 0 {
                let acc = F::from(per_class_correct[class]).unwrap()
                    / F::from(per_class_total[class]).unwrap();
                per_class_accuracies.push(acc);
            } else {
                per_class_accuracies.push(F::zero());
            }
        }

        let balanced_accuracy =
            per_class_accuracies.iter().copied().sum::<F>() / F::from(n_classes).unwrap();

        Ok(LinearProbingResult {
            overall_accuracy,
            balanced_accuracy,
            per_class_accuracies,
            n_classes,
            n_test_samples: test_labels.len(),
        })
    }

    /// Compute representation rank (effective dimensionality)
    pub fn representation_rank(
        &self,
        representations: &Array2<F>,
        threshold: F,
    ) -> Result<RepresentationRankResult<F>> {
        if representations.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Empty representations".to_string(),
            ));
        }

        // Compute covariance matrix
        let cov = self.compute_covariance_matrix(representations)?;

        // Compute eigenvalues (simplified approximation using diagonal)
        let mut eigenvalues: Vec<F> = (0..cov.nrows()).map(|i| cov[[i, i]]).collect();
        eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Compute total variance
        let total_variance: F = eigenvalues.iter().copied().sum();

        // Find effective rank
        let mut cumulative_variance = F::zero();
        let mut effective_rank = 0;

        for &eigenval in &eigenvalues {
            cumulative_variance = cumulative_variance + eigenval;
            effective_rank += 1;

            if cumulative_variance / total_variance >= threshold {
                break;
            }
        }

        // Compute participation ratio
        let sum_eigenvals: F = eigenvalues.iter().copied().sum();
        let sum_eigenvals_sq: F = eigenvalues.iter().map(|&x| x * x).sum();
        let participation_ratio = if sum_eigenvals_sq > F::zero() {
            (sum_eigenvals * sum_eigenvals) / sum_eigenvals_sq
        } else {
            F::zero()
        };

        Ok(RepresentationRankResult {
            effective_rank,
            participation_ratio,
            eigenvalues,
            total_variance,
            explained_variance_ratio: cumulative_variance / total_variance,
        })
    }

    /// Compute clustering-based evaluation (simplified NMI)
    pub fn clustering_normalized_mutual_information(
        &self,
        representations: &Array2<F>,
        true_labels: &Array1<usize>,
        n_clusters: usize,
    ) -> Result<ClusteringResult<F>> {
        if representations.nrows() != true_labels.len() {
            return Err(MetricsError::InvalidInput(
                "Mismatched representations and labels".to_string(),
            ));
        }

        if representations.is_empty() {
            return Err(MetricsError::InvalidInput(
                "Empty representations".to_string(),
            ));
        }

        // Perform simple k-means clustering (simplified centroid-based)
        let cluster_assignments = self.simple_kmeans(representations, n_clusters)?;

        // Compute normalized mutual information
        let nmi = self.compute_normalized_mutual_information(true_labels, &cluster_assignments)?;

        // Compute adjusted rand index (simplified)
        let ari = self.compute_adjusted_rand_index(true_labels, &cluster_assignments)?;

        // Compute silhouette score
        let silhouette = self.compute_silhouette_score(representations, &cluster_assignments)?;

        Ok(ClusteringResult {
            normalized_mutual_information: nmi,
            adjusted_rand_index: ari,
            silhouette_score: silhouette,
            cluster_assignments,
            n_clusters,
        })
    }

    /// Compute Euclidean distance
    fn euclidean_distance(&self, a: &Array1<F>, b: &Array1<F>) -> Result<F> {
        if a.len() != b.len() {
            return Err(MetricsError::InvalidInput(
                "Vector dimension mismatch".to_string(),
            ));
        }

        let distance_sq: F = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let diff = x - y;
                diff * diff
            })
            .sum();

        Ok(distance_sq.sqrt())
    }

    /// Compute covariance matrix
    fn compute_covariance_matrix(&self, data: &Array2<F>) -> Result<Array2<F>> {
        let n_samples = data.nrows();
        let n_features = data.ncols();

        if n_samples < 2 {
            return Err(MetricsError::InvalidInput(
                "Need at least 2 samples".to_string(),
            ));
        }

        // Center the data
        let mean = data.mean_axis(Axis(0)).unwrap();
        let centered = data - &mean.insert_axis(Axis(0));

        // Compute covariance
        let mut cov = Array2::zeros((n_features, n_features));

        for i in 0..n_features {
            for j in i..n_features {
                let mut sum = F::zero();
                for k in 0..n_samples {
                    sum = sum + centered[[k, i]] * centered[[k, j]];
                }
                let cov_val = sum / F::from(n_samples - 1).unwrap();
                cov[[i, j]] = cov_val;
                if i != j {
                    cov[[j, i]] = cov_val;
                }
            }
        }

        Ok(cov)
    }

    /// Simple k-means clustering
    fn simple_kmeans(&self, data: &Array2<F>, k: usize) -> Result<Vec<usize>> {
        let n_samples = data.nrows();
        let n_features = data.ncols();

        if k == 0 || k > n_samples {
            return Err(MetricsError::InvalidInput(
                "Invalid number of clusters".to_string(),
            ));
        }

        // Initialize centroids (use first k samples)
        let mut centroids = Array2::zeros((k, n_features));
        for i in 0..k {
            for j in 0..n_features {
                centroids[[i, j]] = data[[i % n_samples, j]];
            }
        }

        let mut assignments = vec![0; n_samples];
        let max_iterations = 100;

        for _ in 0..max_iterations {
            let mut changed = false;

            // Assign points to nearest centroids
            for i in 0..n_samples {
                let mut best_distance = F::infinity();
                let mut best_cluster = 0;

                for j in 0..k {
                    let distance = self.euclidean_distance(
                        &data.row(i).to_owned(),
                        &centroids.row(j).to_owned(),
                    )?;

                    if distance < best_distance {
                        best_distance = distance;
                        best_cluster = j;
                    }
                }

                if assignments[i] != best_cluster {
                    assignments[i] = best_cluster;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // Update centroids
            let mut cluster_counts = vec![0; k];
            centroids.fill(F::zero());

            for i in 0..n_samples {
                let cluster = assignments[i];
                cluster_counts[cluster] += 1;

                for j in 0..n_features {
                    centroids[[cluster, j]] = centroids[[cluster, j]] + data[[i, j]];
                }
            }

            // Normalize centroids
            for i in 0..k {
                if cluster_counts[i] > 0 {
                    let count = F::from(cluster_counts[i]).unwrap();
                    for j in 0..n_features {
                        centroids[[i, j]] = centroids[[i, j]] / count;
                    }
                }
            }
        }

        Ok(assignments)
    }

    /// Compute normalized mutual information
    fn compute_normalized_mutual_information(
        &self,
        true_labels: &Array1<usize>,
        pred_labels: &[usize],
    ) -> Result<F> {
        if true_labels.len() != pred_labels.len() {
            return Err(MetricsError::InvalidInput(
                "Label length mismatch".to_string(),
            ));
        }

        let n = true_labels.len();
        if n == 0 {
            return Ok(F::zero());
        }

        // Build contingency table
        let max_true = *true_labels.iter().max().unwrap_or(&0) + 1;
        let max_pred = *pred_labels.iter().max().unwrap_or(&0) + 1;

        let mut contingency = vec![vec![0; max_pred]; max_true];
        for i in 0..n {
            contingency[true_labels[i]][pred_labels[i]] += 1;
        }

        // Compute marginals
        let mut true_marginal = vec![0; max_true];
        let mut pred_marginal = vec![0; max_pred];

        for i in 0..max_true {
            for j in 0..max_pred {
                true_marginal[i] += contingency[i][j];
                pred_marginal[j] += contingency[i][j];
            }
        }

        // Compute mutual information
        let mut mi = F::zero();
        for i in 0..max_true {
            for j in 0..max_pred {
                if contingency[i][j] > 0 {
                    let p_ij = F::from(contingency[i][j]).unwrap() / F::from(n).unwrap();
                    let p_i = F::from(true_marginal[i]).unwrap() / F::from(n).unwrap();
                    let p_j = F::from(pred_marginal[j]).unwrap() / F::from(n).unwrap();

                    if p_i > F::zero() && p_j > F::zero() {
                        mi = mi + p_ij * (p_ij / (p_i * p_j)).ln();
                    }
                }
            }
        }

        // Compute entropies for normalization
        let mut h_true = F::zero();
        let mut h_pred = F::zero();

        for i in 0..max_true {
            if true_marginal[i] > 0 {
                let p_i = F::from(true_marginal[i]).unwrap() / F::from(n).unwrap();
                h_true = h_true - p_i * p_i.ln();
            }
        }

        for j in 0..max_pred {
            if pred_marginal[j] > 0 {
                let p_j = F::from(pred_marginal[j]).unwrap() / F::from(n).unwrap();
                h_pred = h_pred - p_j * p_j.ln();
            }
        }

        // Normalize
        let denominator = (h_true + h_pred) / F::from(2.0).unwrap();
        if denominator > F::zero() {
            Ok(mi / denominator)
        } else {
            Ok(F::zero())
        }
    }

    /// Compute adjusted rand index (simplified)
    fn compute_adjusted_rand_index(
        &self,
        true_labels: &Array1<usize>,
        pred_labels: &[usize],
    ) -> Result<F> {
        if true_labels.len() != pred_labels.len() {
            return Err(MetricsError::InvalidInput(
                "Label length mismatch".to_string(),
            ));
        }

        let n = true_labels.len();
        if n == 0 {
            return Ok(F::zero());
        }

        // Count agreements
        let mut agreements = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                let same_true = true_labels[i] == true_labels[j];
                let same_pred = pred_labels[i] == pred_labels[j];

                if same_true == same_pred {
                    agreements += 1;
                }
            }
        }

        let total_pairs = n * (n - 1) / 2;
        if total_pairs == 0 {
            return Ok(F::zero());
        }

        // Simplified ARI (just agreement ratio)
        Ok(F::from(agreements).unwrap() / F::from(total_pairs).unwrap())
    }

    /// Compute silhouette score
    fn compute_silhouette_score(
        &self,
        data: &Array2<F>,
        cluster_assignments: &[usize],
    ) -> Result<F> {
        let n_samples = data.nrows();
        if n_samples != cluster_assignments.len() {
            return Err(MetricsError::InvalidInput(
                "Data and assignments length mismatch".to_string(),
            ));
        }

        if n_samples < 2 {
            return Ok(F::zero());
        }

        let mut total_silhouette = F::zero();
        let mut valid_samples = 0;

        for i in 0..n_samples {
            let cluster_i = cluster_assignments[i];

            // Compute average intra-cluster distance
            let mut intra_distance = F::zero();
            let mut intra_count = 0;

            for j in 0..n_samples {
                if i != j && cluster_assignments[j] == cluster_i {
                    let distance =
                        self.euclidean_distance(&data.row(i).to_owned(), &data.row(j).to_owned())?;
                    intra_distance = intra_distance + distance;
                    intra_count += 1;
                }
            }

            if intra_count > 0 {
                intra_distance = intra_distance / F::from(intra_count).unwrap();
            }

            // Compute minimum average inter-cluster distance
            let mut min_inter_distance = F::infinity();
            let max_cluster = *cluster_assignments.iter().max().unwrap_or(&0);

            for other_cluster in 0..=max_cluster {
                if other_cluster != cluster_i {
                    let mut inter_distance = F::zero();
                    let mut inter_count = 0;

                    for j in 0..n_samples {
                        if cluster_assignments[j] == other_cluster {
                            let distance = self.euclidean_distance(
                                &data.row(i).to_owned(),
                                &data.row(j).to_owned(),
                            )?;
                            inter_distance = inter_distance + distance;
                            inter_count += 1;
                        }
                    }

                    if inter_count > 0 {
                        inter_distance = inter_distance / F::from(inter_count).unwrap();
                        min_inter_distance = min_inter_distance.min(inter_distance);
                    }
                }
            }

            // Compute silhouette for this sample
            if min_inter_distance != F::infinity() {
                let max_distance = intra_distance.max(min_inter_distance);
                if max_distance > F::zero() {
                    let silhouette = (min_inter_distance - intra_distance) / max_distance;
                    total_silhouette = total_silhouette + silhouette;
                    valid_samples += 1;
                }
            }
        }

        if valid_samples > 0 {
            Ok(total_silhouette / F::from(valid_samples).unwrap())
        } else {
            Ok(F::zero())
        }
    }
}
