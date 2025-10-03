//! Cross-validation and evaluation methods for clustering algorithms
//!
//! This module provides cross-validation functionality for different clustering
//! algorithms and metric calculation methods for hyperparameter optimization.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::collections::HashMap;
use std::fmt::Debug;

use crate::advanced::{
    adaptive_online_clustering, quantum_kmeans, rl_clustering, AdaptiveOnlineConfig, QuantumConfig,
    RLClusteringConfig,
};
use crate::affinity::{affinity_propagation, AffinityPropagationOptions};
use crate::birch::{birch, BirchOptions};
use crate::density::{dbscan, optics};
use crate::error::{ClusteringError, Result};
use crate::gmm::{gaussian_mixture, CovarianceType, GMMInit, GMMOptions};
use crate::hierarchy::linkage;
use crate::meanshift::mean_shift;
use crate::metrics::{calinski_harabasz_score, davies_bouldin_score, silhouette_score};
use crate::spectral::{spectral_clustering, AffinityMode, SpectralClusteringOptions};
use crate::stability::OptimalKSelector;
use crate::vq::{kmeans, kmeans2};

use super::config::{CVStrategy, EvaluationMetric, TuningConfig};

/// Cross-validation evaluator for clustering algorithms
pub struct ClusteringEvaluator<F: Float> {
    config: TuningConfig,
    phantom: std::marker::PhantomData<F>,
}

impl<
        F: Float
            + FromPrimitive
            + Debug
            + 'static
            + std::iter::Sum
            + std::fmt::Display
            + Send
            + Sync
            + scirs2_core::ndarray::ScalarOperand
            + std::ops::AddAssign
            + std::ops::SubAssign
            + std::ops::MulAssign
            + std::ops::DivAssign
            + std::ops::RemAssign
            + PartialOrd,
    > ClusteringEvaluator<F>
where
    f64: From<F>,
{
    /// Create new clustering evaluator
    pub fn new(config: TuningConfig) -> Self {
        Self {
            config,
            phantom: std::marker::PhantomData,
        }
    }

    /// Cross-validate K-means clustering
    pub fn cross_validate_kmeans(
        &self,
        data: ArrayView2<F>,
        k: usize,
        max_iter: Option<usize>,
        tol: Option<f64>,
        seed: Option<u64>,
    ) -> Result<Vec<f64>> {
        let mut scores = Vec::new();
        let n_samples = data.shape()[0];

        match self.config.cv_config.strategy {
            CVStrategy::KFold => {
                let fold_size = n_samples / self.config.cv_config.n_folds;

                for fold in 0..self.config.cv_config.n_folds {
                    let start_idx = fold * fold_size;
                    let end_idx = if fold == self.config.cv_config.n_folds - 1 {
                        n_samples
                    } else {
                        (fold + 1) * fold_size
                    };

                    // Create train/test split
                    let mut train_indices = Vec::new();
                    let mut test_indices = Vec::new();

                    for i in 0..n_samples {
                        if i >= start_idx && i < end_idx {
                            test_indices.push(i);
                        } else {
                            train_indices.push(i);
                        }
                    }

                    if train_indices.is_empty() || test_indices.is_empty() {
                        continue;
                    }

                    // Extract training data
                    let train_data = self.extract_subset(data, &train_indices)?;

                    // Run K-means on training data
                    match kmeans2(
                        train_data.view(),
                        k,
                        Some(max_iter.unwrap_or(100)),
                        tol.map(|t| F::from(t).unwrap()),
                        None,
                        None,
                        Some(false),
                        seed,
                    ) {
                        Ok((centroids, labels)) => {
                            // Calculate score based on metric
                            let score = self.calculate_metric_score(
                                train_data.view(),
                                &labels.mapv(|x| x),
                                Some(&centroids),
                            )?;
                            scores.push(score);
                        }
                        Err(_) => {
                            // Skip failed runs
                            continue;
                        }
                    }
                }
            }
            _ => {
                // For other CV strategies, implement similar logic
                // For now, just do a single evaluation
                match kmeans2(
                    data,
                    k,
                    Some(max_iter.unwrap_or(100)),
                    tol.map(|t| F::from(t).unwrap()),
                    None,
                    None,
                    Some(false),
                    seed,
                ) {
                    Ok((centroids, labels)) => {
                        let score = self.calculate_metric_score(
                            data,
                            &labels.mapv(|x| x),
                            Some(&centroids),
                        )?;
                        scores.push(score);
                    }
                    Err(_) => {
                        scores.push(f64::NEG_INFINITY);
                    }
                }
            }
        }

        if scores.is_empty() {
            scores.push(f64::NEG_INFINITY);
        }

        Ok(scores)
    }

    /// Cross-validate DBSCAN clustering
    pub fn cross_validate_dbscan(
        &self,
        data: ArrayView2<F>,
        eps: f64,
        min_samples: usize,
    ) -> Result<Vec<f64>> {
        let mut scores = Vec::new();

        // For DBSCAN, we typically don't use cross-validation in the traditional sense
        // since it's not a predictive model. Instead, we evaluate on the full dataset.
        let data_f64 = data.mapv(|x| x.to_f64().unwrap_or(0.0));

        match dbscan(data_f64.view(), eps, min_samples, None) {
            Ok(labels) => {
                // Convert i32 labels to usize (DBSCAN returns -1 for noise, convert to max value)
                let labels_usize = labels.mapv(|x| if x < 0 { usize::MAX } else { x as usize });
                let score = self.calculate_metric_score(data, &labels_usize, None)?;
                scores.push(score);
            }
            Err(_) => {
                scores.push(f64::NEG_INFINITY);
            }
        }

        Ok(scores)
    }

    /// Cross-validate OPTICS clustering
    pub fn cross_validate_optics(
        &self,
        data: ArrayView2<F>,
        min_samples: usize,
        max_eps: Option<F>,
    ) -> Result<Vec<f64>> {
        let n_samples = data.nrows();
        let n_folds = self.config.cv_config.n_folds.min(n_samples);
        let fold_size = n_samples / n_folds;

        let mut scores = Vec::new();

        for fold in 0..n_folds {
            let start_idx = fold * fold_size;
            let end_idx = if fold == n_folds - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };

            // Create training data (all except current fold)
            let train_indices: Vec<usize> = (0..start_idx).chain(end_idx..n_samples).collect();
            let train_data = data.select(scirs2_core::ndarray::Axis(0), &train_indices);

            // Run OPTICS clustering
            match optics(train_data.view(), min_samples, max_eps, None) {
                Ok(result) => {
                    // Extract cluster labels from OPTICS result
                    let cluster_labels = result;

                    if cluster_labels.iter().all(|&label| label == -1) {
                        // No clusters found
                        scores.push(f64::NEG_INFINITY);
                        continue;
                    }

                    // Convert to usize labels for metric calculation
                    let n_clusters =
                        (*cluster_labels.iter().max().unwrap_or(&-1i32) + 1i32) as usize;
                    if n_clusters < 2usize {
                        scores.push(f64::NEG_INFINITY);
                        continue;
                    }

                    let labels: Vec<usize> = cluster_labels
                        .iter()
                        .map(|&label| {
                            if label == -1i32 {
                                0usize
                            } else {
                                (label as usize) + 1usize
                            }
                        })
                        .collect();
                    let labels_array = Array1::from_vec(labels);

                    // Calculate metric score
                    let score =
                        self.calculate_metric_score(train_data.view(), &labels_array, None)?;
                    scores.push(score);
                }
                Err(_) => {
                    scores.push(f64::NEG_INFINITY);
                }
            }
        }

        Ok(scores)
    }

    /// Cross-validate Spectral clustering
    pub fn cross_validate_spectral(
        &self,
        data: ArrayView2<F>,
        n_clusters: usize,
        n_neighbors: usize,
        gamma: F,
        max_iter: usize,
    ) -> Result<Vec<f64>> {
        let n_samples = data.nrows();
        let n_folds = self.config.cv_config.n_folds.min(n_samples);
        let fold_size = n_samples / n_folds;

        let mut scores = Vec::new();

        for fold in 0..n_folds {
            let start_idx = fold * fold_size;
            let end_idx = if fold == n_folds - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };

            let train_indices: Vec<usize> = (0..start_idx).chain(end_idx..n_samples).collect();
            let train_data = data.select(scirs2_core::ndarray::Axis(0), &train_indices);

            // Create spectral clustering options
            let options = SpectralClusteringOptions {
                affinity: AffinityMode::RBF,
                n_neighbors,
                gamma,
                normalized_laplacian: true,
                max_iter,
                n_init: 1,
                tol: F::from(1e-4).unwrap(),
                random_seed: None,
                eigen_solver: "arpack".to_string(),
                auto_n_clusters: false,
            };

            match spectral_clustering(train_data.view(), n_clusters, Some(options)) {
                Ok((_, labels)) => {
                    let score = self.calculate_metric_score(train_data.view(), &labels, None)?;
                    scores.push(score);
                }
                Err(_) => {
                    scores.push(f64::NEG_INFINITY);
                }
            }
        }

        Ok(scores)
    }

    /// Cross-validate Affinity Propagation clustering
    pub fn cross_validate_affinity_propagation(
        &self,
        data: ArrayView2<F>,
        damping: F,
        max_iter: usize,
        convergence_iter: usize,
    ) -> Result<Vec<f64>> {
        let n_samples = data.nrows();
        let n_folds = self.config.cv_config.n_folds.min(n_samples);
        let fold_size = n_samples / n_folds;

        let mut scores = Vec::new();

        for fold in 0..n_folds {
            let start_idx = fold * fold_size;
            let end_idx = if fold == n_folds - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };

            let train_indices: Vec<usize> = (0..start_idx).chain(end_idx..n_samples).collect();
            let train_data = data.select(scirs2_core::ndarray::Axis(0), &train_indices);

            // Create affinity propagation options
            let options = AffinityPropagationOptions {
                damping,
                max_iter,
                convergence_iter,
                preference: None, // Use default (median of similarities)
                affinity: "euclidean".to_string(),
                max_affinity_iterations: 10,
            };

            match affinity_propagation(train_data.view(), false, Some(options)) {
                Ok((_, labels)) => {
                    // Convert i32 labels to usize
                    let usize_labels: Vec<usize> = labels.iter().map(|&x| x as usize).collect();
                    let labels_array = Array1::from_vec(usize_labels);

                    let score =
                        self.calculate_metric_score(train_data.view(), &labels_array, None)?;
                    scores.push(score);
                }
                Err(_) => {
                    scores.push(f64::NEG_INFINITY);
                }
            }
        }

        Ok(scores)
    }

    /// Cross-validate BIRCH clustering
    pub fn cross_validate_birch(
        &self,
        data: ArrayView2<F>,
        branching_factor: usize,
        threshold: F,
    ) -> Result<Vec<f64>> {
        let n_samples = data.nrows();
        let n_folds = self.config.cv_config.n_folds.min(n_samples);
        let fold_size = n_samples / n_folds;

        let mut scores = Vec::new();

        for fold in 0..n_folds {
            let start_idx = fold * fold_size;
            let end_idx = if fold == n_folds - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };

            let train_indices: Vec<usize> = (0..start_idx).chain(end_idx..n_samples).collect();
            let train_data = data.select(scirs2_core::ndarray::Axis(0), &train_indices);

            // Create BIRCH options
            let options = BirchOptions {
                branching_factor,
                threshold,
                n_clusters: None, // Use all clusters found
            };

            match birch(train_data.view(), options) {
                Ok((_, labels)) => {
                    // Convert i32 labels to usize
                    let usize_labels: Vec<usize> = labels.iter().map(|&x| x as usize).collect();
                    let labels_array = Array1::from_vec(usize_labels);

                    let score =
                        self.calculate_metric_score(train_data.view(), &labels_array, None)?;
                    scores.push(score);
                }
                Err(_) => {
                    scores.push(f64::NEG_INFINITY);
                }
            }
        }

        Ok(scores)
    }

    /// Cross-validate GMM clustering
    pub fn cross_validate_gmm(
        &self,
        data: ArrayView2<F>,
        n_components: usize,
        max_iter: usize,
        tol: F,
        reg_covar: F,
    ) -> Result<Vec<f64>> {
        let n_samples = data.nrows();
        let n_folds = self.config.cv_config.n_folds.min(n_samples);
        let fold_size = n_samples / n_folds;

        let mut scores = Vec::new();

        for fold in 0..n_folds {
            let start_idx = fold * fold_size;
            let end_idx = if fold == n_folds - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };

            let train_indices: Vec<usize> = (0..start_idx).chain(end_idx..n_samples).collect();
            let train_data = data.select(scirs2_core::ndarray::Axis(0), &train_indices);

            // Convert to f64 for GMM
            let train_data_f64 = train_data.mapv(|x| x.to_f64().unwrap_or(0.0));

            // Create GMM options
            let options = GMMOptions {
                n_components,
                covariance_type: CovarianceType::Full,
                tol: tol.to_f64().unwrap_or(1e-4),
                max_iter,
                n_init: 1,
                init_method: GMMInit::KMeans,
                random_seed: Some(42),
                reg_covar: reg_covar.to_f64().unwrap_or(1e-6),
            };

            match gaussian_mixture(train_data_f64.view(), options) {
                Ok(labels) => {
                    // Convert i32 labels to usize
                    let usize_labels: Vec<usize> = labels.iter().map(|&x| x as usize).collect();
                    let labels_array = Array1::from_vec(usize_labels);

                    let score =
                        self.calculate_metric_score(train_data.view(), &labels_array, None)?;
                    scores.push(score);
                }
                Err(_) => {
                    scores.push(f64::NEG_INFINITY);
                }
            }
        }

        Ok(scores)
    }

    /// Calculate metric score for evaluation
    pub fn calculate_metric_score(
        &self,
        data: ArrayView2<F>,
        labels: &Array1<usize>,
        centroids: Option<&Array2<F>>,
    ) -> Result<f64> {
        let data_f64 = data.mapv(|x| x.to_f64().unwrap_or(0.0));
        let labels_i32 = labels.mapv(|x| x as i32);

        match self.config.metric {
            EvaluationMetric::SilhouetteScore => {
                silhouette_score(data_f64.view(), labels_i32.view())
            }
            EvaluationMetric::DaviesBouldinIndex => {
                davies_bouldin_score(data_f64.view(), labels_i32.view())
            }
            EvaluationMetric::CalinskiHarabaszIndex => {
                calinski_harabasz_score(data_f64.view(), labels_i32.view())
            }
            EvaluationMetric::Inertia => {
                // Calculate within-cluster sum of squares
                if let Some(centroids) = centroids {
                    let centroids_f64 = centroids.mapv(|x| x.to_f64().unwrap_or(0.0));
                    self.calculate_inertia(&data_f64, labels, &centroids_f64)
                } else {
                    Ok(f64::INFINITY) // Invalid for algorithms without centroids
                }
            }
            _ => Ok(0.0), // Placeholder for other metrics
        }
    }

    /// Calculate inertia (within-cluster sum of squares)
    pub fn calculate_inertia(
        &self,
        data: &Array2<f64>,
        labels: &Array1<usize>,
        centroids: &Array2<f64>,
    ) -> Result<f64> {
        let mut total_inertia = 0.0;

        for (i, &label) in labels.iter().enumerate() {
            let mut distance_sq = 0.0;
            for j in 0..data.ncols() {
                let diff = data[[i, j]] - centroids[[label, j]];
                distance_sq += diff * diff;
            }
            total_inertia += distance_sq;
        }

        Ok(total_inertia)
    }

    /// Extract subset of data based on indices
    pub fn extract_subset(&self, data: ArrayView2<F>, indices: &[usize]) -> Result<Array2<F>> {
        let n_features = data.ncols();
        let mut subset = Array2::zeros((indices.len(), n_features));

        for (new_idx, &old_idx) in indices.iter().enumerate() {
            if old_idx < data.nrows() {
                subset.row_mut(new_idx).assign(&data.row(old_idx));
            }
        }

        Ok(subset)
    }

    /// Calculate correlation between two vectors
    pub fn calculate_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return 0.0;
        }

        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let sum_x_sq: f64 = x.iter().map(|a| a * a).sum();
        let sum_y_sq: f64 = y.iter().map(|a| a * a).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x_sq - sum_x * sum_x) * (n * sum_y_sq - sum_y * sum_y)).sqrt();

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }
}
