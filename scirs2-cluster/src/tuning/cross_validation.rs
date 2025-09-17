//! Cross-validation implementations for clustering algorithms
//!
//! This module provides cross-validation methods for different clustering
//! algorithms to evaluate hyperparameter configurations.

use ndarray::{Array1, Array2, ArrayView2, Axis};
use num_traits::{Float, FromPrimitive};
use std::collections::HashMap;
use std::fmt::Debug;

use crate::affinity::{affinity_propagation, AffinityPropagationOptions};
use crate::birch::{birch, BirchOptions};
use crate::density::{dbscan, optics};
use crate::error::{ClusteringError, Result};
use crate::gmm::{gaussian_mixture, CovarianceType, GMMInit, GMMOptions};
use crate::metrics::{calinski_harabasz_score, davies_bouldin_score, silhouette_score};
use crate::spectral::{spectral_clustering, AffinityMode, SpectralClusteringOptions};
use crate::vq::{kmeans, kmeans2};

use super::config::{CVStrategy, CrossValidationConfig, EvaluationMetric};
use super::utilities::calculate_inertia;

use statrs::statistics::Statistics;

/// Cross-validation engine for clustering algorithms
pub struct CrossValidator {
    config: CrossValidationConfig,
}

impl CrossValidator {
    /// Create a new cross-validator with specified configuration
    pub fn new(config: &CrossValidationConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    /// Cross-validate K-means clustering
    pub fn cross_validate_kmeans<F>(
        &self,
        data: ArrayView2<F>,
        k: usize,
        max_iter: Option<usize>,
        tol: Option<f64>,
        seed: Option<u64>,
        metric: &EvaluationMetric,
    ) -> Result<Vec<f64>>
    where
        F: Float
            + FromPrimitive
            + Debug
            + 'static
            + std::iter::Sum
            + std::fmt::Display
            + Send
            + Sync
            + ndarray::ScalarOperand
            + std::ops::AddAssign
            + std::ops::SubAssign
            + std::ops::MulAssign
            + std::ops::DivAssign
            + std::ops::RemAssign
            + PartialOrd,
        f64: From<F>,
    {
        let mut scores = Vec::new();
        let n_samples = data.shape()[0];

        match self.config.strategy {
            CVStrategy::KFold => {
                let fold_size = n_samples / self.config.n_folds;

                for fold in 0..self.config.n_folds {
                    let start_idx = fold * fold_size;
                    let end_idx = if fold == self.config.n_folds - 1 {
                        n_samples
                    } else {
                        (fold + 1) * fold_size
                    };

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

                    let train_data = extract_subset(data, &train_indices)?;

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
                            let score = calculate_metric_score(
                                train_data.view(),
                                &labels.mapv(|x| x),
                                Some(&centroids),
                                metric,
                            )?;
                            scores.push(score);
                        }
                        Err(_) => continue,
                    }
                }
            }
            _ => {
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
                        let score = calculate_metric_score(
                            data,
                            &labels.mapv(|x| x),
                            Some(&centroids),
                            metric,
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
    pub fn cross_validate_dbscan<F>(
        &self,
        data: ArrayView2<F>,
        eps: f64,
        min_samples: usize,
        metric: &EvaluationMetric,
    ) -> Result<Vec<f64>>
    where
        F: Float + FromPrimitive + Debug + Send + Sync + ndarray::ScalarOperand,
        f64: From<F>,
    {
        let mut scores = Vec::new();
        let data_f64 = data.mapv(|x| x.to_f64().unwrap_or(0.0));

        match dbscan(data_f64.view(), eps, min_samples, None) {
            Ok(labels) => {
                let labels_usize = labels.mapv(|x| if x < 0 { usize::MAX } else { x as usize });
                let score = calculate_metric_score(data, &labels_usize, None, metric)?;
                scores.push(score);
            }
            Err(_) => {
                scores.push(f64::NEG_INFINITY);
            }
        }

        Ok(scores)
    }

    /// Cross-validate OPTICS clustering
    pub fn cross_validate_optics<F>(
        &self,
        data: ArrayView2<F>,
        min_samples: usize,
        max_eps: Option<F>,
        metric: &EvaluationMetric,
    ) -> Result<Vec<f64>>
    where
        F: Float + FromPrimitive + Debug + Send + Sync + ndarray::ScalarOperand,
        f64: From<F>,
    {
        let n_samples = data.nrows();
        let n_folds = self.config.n_folds.min(n_samples);
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
            let train_data = data.select(Axis(0), &train_indices);

            match optics(train_data.view(), min_samples, max_eps, None) {
                Ok(result) => {
                    let cluster_labels = result;

                    if cluster_labels.iter().all(|&label| label == -1) {
                        scores.push(f64::NEG_INFINITY);
                        continue;
                    }

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

                    let score =
                        calculate_metric_score(train_data.view(), &labels_array, None, metric)?;
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
    pub fn cross_validate_spectral<F>(
        &self,
        data: ArrayView2<F>,
        n_clusters: usize,
        n_neighbors: usize,
        gamma: F,
        max_iter: usize,
        metric: &EvaluationMetric,
    ) -> Result<Vec<f64>>
    where
        F: Float
            + FromPrimitive
            + Debug
            + Send
            + Sync
            + ndarray::ScalarOperand
            + std::ops::MulAssign
            + std::ops::DivAssign
            + std::ops::RemAssign
            + std::fmt::Display
            + std::iter::Sum
            + std::ops::AddAssign
            + std::ops::SubAssign,
        f64: From<F>,
    {
        let n_samples = data.nrows();
        let n_folds = self.config.n_folds.min(n_samples);
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
            let train_data = data.select(Axis(0), &train_indices);

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
                    let score = calculate_metric_score(train_data.view(), &labels, None, metric)?;
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
    pub fn cross_validate_affinity_propagation<F>(
        &self,
        data: ArrayView2<F>,
        damping: F,
        max_iter: usize,
        convergence_iter: usize,
        metric: &EvaluationMetric,
    ) -> Result<Vec<f64>>
    where
        F: Float + FromPrimitive + Debug + Send + Sync + ndarray::ScalarOperand,
        f64: From<F>,
    {
        let n_samples = data.nrows();
        let n_folds = self.config.n_folds.min(n_samples);
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
            let train_data = data.select(Axis(0), &train_indices);

            let options = AffinityPropagationOptions {
                damping,
                max_iter,
                convergence_iter,
                preference: None,
                affinity: "euclidean".to_string(),
                max_affinity_iterations: 10,
            };

            match affinity_propagation(train_data.view(), false, Some(options)) {
                Ok((_, labels)) => {
                    let usize_labels: Vec<usize> = labels.iter().map(|&x| x as usize).collect();
                    let labels_array = Array1::from_vec(usize_labels);

                    let score =
                        calculate_metric_score(train_data.view(), &labels_array, None, metric)?;
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
    pub fn cross_validate_birch<F>(
        &self,
        data: ArrayView2<F>,
        branching_factor: usize,
        threshold: F,
        metric: &EvaluationMetric,
    ) -> Result<Vec<f64>>
    where
        F: Float + FromPrimitive + Debug + Send + Sync + ndarray::ScalarOperand,
        f64: From<F>,
    {
        let n_samples = data.nrows();
        let n_folds = self.config.n_folds.min(n_samples);
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
            let train_data = data.select(Axis(0), &train_indices);

            let options = BirchOptions {
                branching_factor,
                threshold,
                n_clusters: None,
            };

            match birch(train_data.view(), options) {
                Ok((_, labels)) => {
                    let usize_labels: Vec<usize> = labels.iter().map(|&x| x as usize).collect();
                    let labels_array = Array1::from_vec(usize_labels);

                    let score =
                        calculate_metric_score(train_data.view(), &labels_array, None, metric)?;
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
    pub fn cross_validate_gmm<F>(
        &self,
        data: ArrayView2<F>,
        n_components: usize,
        max_iter: usize,
        tol: F,
        reg_covar: F,
        metric: &EvaluationMetric,
    ) -> Result<Vec<f64>>
    where
        F: Float + FromPrimitive + Debug + Send + Sync + ndarray::ScalarOperand,
        f64: From<F>,
    {
        let n_samples = data.nrows();
        let n_folds = self.config.n_folds.min(n_samples);
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
            let train_data = data.select(Axis(0), &train_indices);
            let train_data_f64 = train_data.mapv(|x| x.to_f64().unwrap_or(0.0));

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
                    let usize_labels: Vec<usize> = labels.iter().map(|&x| x as usize).collect();
                    let labels_array = Array1::from_vec(usize_labels);

                    let score =
                        calculate_metric_score(train_data.view(), &labels_array, None, metric)?;
                    scores.push(score);
                }
                Err(_) => {
                    scores.push(f64::NEG_INFINITY);
                }
            }
        }

        Ok(scores)
    }
}

/// Extract subset of data based on indices
fn extract_subset<F>(data: ArrayView2<F>, indices: &[usize]) -> Result<Array2<F>>
where
    F: Clone + num_traits::Zero,
{
    let n_features = data.ncols();
    let mut subset = Array2::zeros((indices.len(), n_features));

    for (new_idx, &old_idx) in indices.iter().enumerate() {
        if old_idx < data.nrows() {
            subset.row_mut(new_idx).assign(&data.row(old_idx));
        }
    }

    Ok(subset)
}

/// Calculate metric score for evaluation
fn calculate_metric_score<F>(
    data: ArrayView2<F>,
    labels: &Array1<usize>,
    centroids: Option<&Array2<F>>,
    metric: &EvaluationMetric,
) -> Result<f64>
where
    F: Float + FromPrimitive + Debug + Send + Sync + ndarray::ScalarOperand,
    f64: From<F>,
{
    let data_f64 = data.mapv(|x| x.to_f64().unwrap_or(0.0));
    let labels_i32 = labels.mapv(|x| x as i32);

    match metric {
        EvaluationMetric::SilhouetteScore => silhouette_score(data_f64.view(), labels_i32.view()),
        EvaluationMetric::DaviesBouldinIndex => {
            davies_bouldin_score(data_f64.view(), labels_i32.view())
        }
        EvaluationMetric::CalinskiHarabaszIndex => {
            calinski_harabasz_score(data_f64.view(), labels_i32.view())
        }
        EvaluationMetric::Inertia => {
            if let Some(centroids) = centroids {
                let centroids_f64 = centroids.mapv(|x| x.to_f64().unwrap_or(0.0));
                calculate_inertia(&data_f64, labels, &centroids_f64)
            } else {
                Ok(f64::INFINITY)
            }
        }
        _ => Ok(0.0),
    }
}
