//! Stability-based clustering validation methods
//!
//! This module provides methods to analyze clustering stability through various
//! resampling techniques including bootstrap, subsampling, and noise perturbation.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use scirs2_core::random::seq::SliceRandom;
use scirs2_core::random::Rng;
use std::collections::HashMap;
use std::fmt::{Debug, Display};

use crate::error::{ClusteringError, Result};
use crate::metrics::silhouette::silhouette_score;

/// Configuration for advanced stability analysis
#[derive(Debug, Clone)]
pub struct StabilityConfig {
    /// Number of bootstrap iterations
    pub n_bootstrap: usize,
    /// Number of subsampling iterations
    pub n_subsamples: usize,
    /// Ratio of samples to use in subsampling
    pub subsample_ratio: f64,
    /// Standard deviation of noise for perturbation analysis
    pub noise_perturbation: f64,
    /// Whether to perform feature subsampling
    pub feature_subsampling: bool,
    /// Whether to perform temporal analysis
    pub temporal_analysis: bool,
}

impl Default for StabilityConfig {
    fn default() -> Self {
        Self {
            n_bootstrap: 100,
            n_subsamples: 50,
            subsample_ratio: 0.8,
            noise_perturbation: 0.01,
            feature_subsampling: true,
            temporal_analysis: false,
        }
    }
}

/// Comprehensive stability analysis result
#[derive(Debug, Clone)]
pub struct StabilityResult<F: Float> {
    /// Bootstrap stability score
    pub bootstrap_stability: F,
    /// Subsample stability score
    pub subsample_stability: F,
    /// Noise stability score
    pub noise_stability: F,
    /// Feature stability score
    pub feature_stability: F,
    /// Connectivity stability score
    pub connectivity_stability: F,
    /// Per-cluster persistence scores
    pub cluster_persistence: Vec<F>,
    /// Stability trend over iterations
    pub stability_trend: Vec<F>,
    /// Confidence intervals (lower, median, upper)
    pub confidence_intervals: (F, F, F),
}

/// Bootstrap confidence interval for clustering metrics.
///
/// This function computes confidence intervals for clustering quality metrics
/// using bootstrap resampling. It repeatedly samples data with replacement
/// and computes the metric to estimate the distribution.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `labels` - Cluster assignments for each sample
/// * `confidence_level` - Confidence level (e.g., 0.95 for 95% CI)
/// * `n_bootstrap` - Number of bootstrap iterations
///
/// # Returns
/// * `Result<(F, F, F)>` - Tuple of (lower_bound, mean_score, upper_bound)
///
/// # Example
/// ```rust
/// use scirs2_core::ndarray::{Array1, Array2};
/// use scirs2_cluster::metrics::bootstrap_confidence_interval;
///
/// let data = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 1.0, 10.0, 10.0, 11.0, 11.0]).unwrap();
/// let labels = Array1::from_vec(vec![0, 0, 1, 1]);
/// let (lower, mean, upper) = bootstrap_confidence_interval(
///     data.view(), labels.view(), 0.95, 100
/// ).unwrap();
/// ```
pub fn bootstrap_confidence_interval<F>(
    data: ArrayView2<F>,
    labels: ArrayView1<i32>,
    confidence_level: f64,
    n_bootstrap: usize,
) -> Result<(F, F, F)>
where
    F: Float + FromPrimitive + Debug + PartialOrd + Copy + std::iter::Sum + Display + 'static,
{
    if confidence_level <= 0.0 || confidence_level >= 1.0 {
        return Err(ClusteringError::InvalidInput(
            "Confidence level must be between 0 and 1".to_string(),
        ));
    }

    let n_samples = data.shape()[0];
    let mut rng = scirs2_core::random::rng();
    let mut bootstrap_scores = Vec::new();

    // Perform bootstrap resampling
    for _iter in 0..n_bootstrap {
        // Create bootstrap sample
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut rng);

        // Sample with replacement
        let bootstrap_indices: Vec<usize> = (0..n_samples)
            .map(|_| indices[rng.random_range(0..n_samples)])
            .collect();

        // Extract bootstrap sample
        let bootstrap_data = data.select(scirs2_core::ndarray::Axis(0), &bootstrap_indices);
        let bootstrap_labels: Vec<i32> = bootstrap_indices.iter().map(|&i| labels[i]).collect();
        let bootstrap_labels_array = Array1::from_vec(bootstrap_labels);

        // Compute metric (using silhouette score as example)
        if let Ok(score) = silhouette_score(bootstrap_data.view(), bootstrap_labels_array.view()) {
            bootstrap_scores.push(score);
        }
    }

    if bootstrap_scores.is_empty() {
        return Err(ClusteringError::ComputationError(
            "No successful bootstrap iterations".to_string(),
        ));
    }

    // Sort scores for percentile calculation
    bootstrap_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Calculate confidence interval
    let alpha = 1.0 - confidence_level;
    let lower_percentile = alpha / 2.0;
    let upper_percentile = 1.0 - alpha / 2.0;

    let lower_idx = (bootstrap_scores.len() as f64 * lower_percentile) as usize;
    let upper_idx = (bootstrap_scores.len() as f64 * upper_percentile) as usize;

    let lower_bound = bootstrap_scores[lower_idx.min(bootstrap_scores.len() - 1)];
    let upper_bound = bootstrap_scores[upper_idx.min(bootstrap_scores.len() - 1)];

    // Calculate mean score
    let mean_score = bootstrap_scores.iter().fold(F::zero(), |acc, &x| acc + x)
        / F::from(bootstrap_scores.len()).unwrap();

    Ok((lower_bound, mean_score, upper_bound))
}

/// Multi-scale clustering stability analysis.
///
/// Performs comprehensive stability analysis using multiple techniques:
/// bootstrap resampling, subsampling, noise perturbation, and feature selection.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `n_clusters` - Number of clusters to use
/// * `config` - Configuration for stability analysis
///
/// # Returns
/// * `Result<StabilityResult<F>>` - Comprehensive stability results
pub fn comprehensive_stability_analysis<F>(
    data: ArrayView2<F>,
    n_clusters: usize,
    config: StabilityConfig,
) -> Result<StabilityResult<F>>
where
    F: Float + FromPrimitive + Debug + PartialOrd + Copy + std::iter::Sum + Display + 'static,
{
    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];

    if n_samples < n_clusters {
        return Err(ClusteringError::InvalidInput(
            "Number of samples must be greater than number of clusters".to_string(),
        ));
    }

    let mut rng = scirs2_core::random::rng();
    let mut bootstrap_scores = Vec::new();
    let mut subsample_scores = Vec::new();
    let mut noise_scores = Vec::new();
    let mut feature_scores = Vec::new();

    // Bootstrap stability analysis
    for _iter in 0..config.n_bootstrap {
        let bootstrap_indices: Vec<usize> = (0..n_samples)
            .map(|_| rng.random_range(0..n_samples))
            .collect();

        let bootstrap_data = data.select(scirs2_core::ndarray::Axis(0), &bootstrap_indices);

        // Perform clustering (simplified k-means approach)
        if let Ok(labels) = simple_kmeans(bootstrap_data.view(), n_clusters) {
            if let Ok(score) = silhouette_score(bootstrap_data.view(), labels.view()) {
                bootstrap_scores.push(score);
            }
        }
    }

    // Subsample stability analysis
    let subsample_size = (n_samples as f64 * config.subsample_ratio) as usize;
    for _iter in 0..config.n_subsamples {
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut rng);
        indices.truncate(subsample_size);

        let subsample_data = data.select(scirs2_core::ndarray::Axis(0), &indices);

        if let Ok(labels) = simple_kmeans(subsample_data.view(), n_clusters) {
            if let Ok(score) = silhouette_score(subsample_data.view(), labels.view()) {
                subsample_scores.push(score);
            }
        }
    }

    // Noise perturbation stability analysis
    for _iter in 0..config.n_bootstrap {
        let mut noisy_data = data.to_owned();

        // Add Gaussian noise
        for i in 0..n_samples {
            for j in 0..n_features {
                let noise =
                    F::from(rng.random_range(-1.0..1.0) * config.noise_perturbation).unwrap();
                noisy_data[[i, j]] = noisy_data[[i, j]] + noise;
            }
        }

        if let Ok(labels) = simple_kmeans(noisy_data.view(), n_clusters) {
            if let Ok(score) = silhouette_score(noisy_data.view(), labels.view()) {
                noise_scores.push(score);
            }
        }
    }

    // Feature subsampling stability (if enabled)
    if config.feature_subsampling && n_features > 1 {
        let feature_subset_size = (n_features as f64 * 0.8) as usize;

        for _iter in 0..config.n_subsamples {
            let mut feature_indices: Vec<usize> = (0..n_features).collect();
            feature_indices.shuffle(&mut rng);
            feature_indices.truncate(feature_subset_size);

            let feature_subset_data = data.select(scirs2_core::ndarray::Axis(1), &feature_indices);

            if let Ok(labels) = simple_kmeans(feature_subset_data.view(), n_clusters) {
                if let Ok(score) = silhouette_score(feature_subset_data.view(), labels.view()) {
                    feature_scores.push(score);
                }
            }
        }
    }

    // Calculate stability metrics
    let bootstrap_stability = calculate_stability(&bootstrap_scores);
    let subsample_stability = calculate_stability(&subsample_scores);
    let noise_stability = calculate_stability(&noise_scores);
    let feature_stability = calculate_stability(&feature_scores);

    // Simplified connectivity stability (placeholder)
    let connectivity_stability = F::from(0.5).unwrap();

    // Calculate confidence intervals from bootstrap scores
    let mut all_scores = bootstrap_scores.clone();
    all_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let confidence_intervals = if !all_scores.is_empty() {
        let len = all_scores.len();
        let lower_idx = (len as f64 * 0.025) as usize;
        let upper_idx = (len as f64 * 0.975) as usize;
        let median_idx = len / 2;

        (
            all_scores[lower_idx.min(len - 1)],
            all_scores[median_idx],
            all_scores[upper_idx.min(len - 1)],
        )
    } else {
        (F::zero(), F::zero(), F::zero())
    };

    Ok(StabilityResult {
        bootstrap_stability,
        subsample_stability,
        noise_stability,
        feature_stability,
        connectivity_stability,
        cluster_persistence: vec![F::from(0.8).unwrap(); n_clusters], // Placeholder
        stability_trend: bootstrap_scores,
        confidence_intervals,
    })
}

/// Calculate stability measure from a collection of scores.
fn calculate_stability<F>(scores: &[F]) -> F
where
    F: Float + FromPrimitive + Debug + Copy + std::iter::Sum,
{
    if scores.is_empty() {
        return F::zero();
    }

    // Calculate coefficient of variation (inverse stability)
    let mean = scores.iter().fold(F::zero(), |acc, &x| acc + x) / F::from(scores.len()).unwrap();

    if mean == F::zero() {
        return F::zero();
    }

    let variance = scores
        .iter()
        .map(|&x| {
            let diff = x - mean;
            diff * diff
        })
        .fold(F::zero(), |acc, x| acc + x)
        / F::from(scores.len()).unwrap();

    let std_dev = variance.sqrt();
    let coefficient_of_variation = std_dev / mean;

    // Return stability as 1 / (1 + CV)
    F::one() / (F::one() + coefficient_of_variation)
}

/// Simplified k-means clustering for stability analysis.
fn simple_kmeans<F>(data: ArrayView2<F>, n_clusters: usize) -> Result<Array1<i32>>
where
    F: Float + FromPrimitive + Debug + PartialOrd + Copy + 'static,
{
    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];

    if n_samples < n_clusters {
        return Err(ClusteringError::InvalidInput(
            "Number of samples must be greater than number of clusters".to_string(),
        ));
    }

    let mut rng = scirs2_core::random::rng();
    let mut centroids = Array2::zeros((n_clusters, n_features));

    // Initialize centroids randomly
    for i in 0..n_clusters {
        let random_idx = rng.random_range(0..n_samples);
        for j in 0..n_features {
            centroids[[i, j]] = data[[random_idx, j]];
        }
    }

    let mut labels = Array1::zeros(n_samples);
    let max_iterations = 100;

    for _iter in 0..max_iterations {
        let mut changed = false;

        // Assign points to closest centroids
        for i in 0..n_samples {
            let mut min_distance = F::infinity();
            let mut best_cluster = 0;

            for k in 0..n_clusters {
                let mut distance = F::zero();
                for j in 0..n_features {
                    let diff = data[[i, j]] - centroids[[k, j]];
                    distance = distance + diff * diff;
                }

                if distance < min_distance {
                    min_distance = distance;
                    best_cluster = k;
                }
            }

            if labels[i] != best_cluster as i32 {
                labels[i] = best_cluster as i32;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Update centroids
        let mut cluster_counts = vec![0; n_clusters];
        centroids.fill(F::zero());

        for i in 0..n_samples {
            let cluster = labels[i] as usize;
            cluster_counts[cluster] += 1;
            for j in 0..n_features {
                centroids[[cluster, j]] = centroids[[cluster, j]] + data[[i, j]];
            }
        }

        for k in 0..n_clusters {
            if cluster_counts[k] > 0 {
                let count = F::from(cluster_counts[k]).unwrap();
                for j in 0..n_features {
                    centroids[[k, j]] = centroids[[k, j]] / count;
                }
            }
        }
    }

    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_bootstrap_confidence_interval() {
        let data = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 1.0, 10.0, 10.0, 11.0, 11.0])
            .unwrap();
        let labels = Array1::from_vec(vec![0, 0, 1, 1]);

        let result = bootstrap_confidence_interval(data.view(), labels.view(), 0.95, 10);
        assert!(result.is_ok());

        let (lower, mean, upper) = result.unwrap();
        assert!(lower <= mean && mean <= upper);
    }

    #[test]
    fn test_comprehensive_stability_analysis() {
        let data = Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 10.0, 10.0, 11.0, 11.0, 12.0, 12.0, 13.0,
                13.0,
            ],
        )
        .unwrap();

        let config = StabilityConfig {
            n_bootstrap: 5,
            n_subsamples: 5,
            ..Default::default()
        };

        let result = comprehensive_stability_analysis(data.view(), 2, config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_simple_kmeans() {
        let data = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 1.0, 10.0, 10.0, 11.0, 11.0])
            .unwrap();

        let labels = simple_kmeans(data.view(), 2);
        assert!(labels.is_ok());

        let labels = labels.unwrap();
        assert_eq!(labels.len(), 4);
    }

    #[test]
    fn test_calculate_stability() {
        let scores = vec![0.8, 0.85, 0.82, 0.88, 0.79];
        let stability = calculate_stability(&scores);
        assert!(stability > 0.0 && stability <= 1.0);
    }
}
