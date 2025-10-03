//! Clustering metrics module
//!
//! This module provides functions for evaluating clustering algorithms, including
//! silhouette score, Davies-Bouldin index, Calinski-Harabasz index, and Adjusted Rand index.
//!
//! ## Internal Metrics
//!
//! Internal metrics assess clustering without external ground truth:
//! - Silhouette score
//! - Davies-Bouldin index
//! - Calinski-Harabasz index
//! - Dunn index
//! - Inter-cluster and intra-cluster distance metrics
//!
//! ## External Metrics
//!
//! External metrics assess clustering compared to ground truth:
//! - Adjusted Rand index
//! - Normalized Mutual Information
//! - Adjusted Mutual Information
//! - Homogeneity, Completeness, V-measure
//! - Fowlkes-Mallows score

pub mod evaluation;
pub mod external_metrics;
mod internal_metrics;

// Existing submodules
pub mod density;
pub mod distance;
pub mod validation;

// Re-export all public items from submodules - be specific with the evaluation module
pub use self::density::*;
pub use self::distance::*;
pub use self::evaluation::{dunn_index_enhanced, elbow_method};
pub use self::external_metrics::*;
pub use self::internal_metrics::*;
pub use self::validation::*;

// Common utility functions that might be used across multiple submodules
use scirs2_core::ndarray::{Array1, Array2, ArrayBase, Data, Dimension, Ix2};
use scirs2_core::numeric::{Float, NumCast};
use std::collections::HashMap;

use crate::error::{MetricsError, Result};

/// Calculate distance between two points
///
/// # Arguments
///
/// * `a` - First point
/// * `b` - Second point
/// * `metric` - Distance metric name ("euclidean", "manhattan", "cosine", etc.)
///
/// # Returns
///
/// * The distance between the points
pub(crate) fn calculate_distance<F>(a: &[F], b: &[F], metric: &str) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug,
{
    if a.len() != b.len() {
        return Err(MetricsError::InvalidInput(format!(
            "Points must have the same dimension, got {} and {}",
            a.len(),
            b.len()
        )));
    }

    match metric.to_lowercase().as_str() {
        "euclidean" => {
            let mut sum = F::zero();
            for (ai, bi) in a.iter().zip(b.iter()) {
                let diff = *ai - *bi;
                sum = sum + diff * diff;
            }
            Ok(sum.sqrt())
        }
        "manhattan" => {
            let mut sum = F::zero();
            for (ai, bi) in a.iter().zip(b.iter()) {
                sum = sum + (*ai - *bi).abs();
            }
            Ok(sum)
        }
        "cosine" => {
            let mut dot = F::zero();
            let mut norm_a = F::zero();
            let mut norm_b = F::zero();

            for (ai, bi) in a.iter().zip(b.iter()) {
                dot = dot + *ai * *bi;
                norm_a = norm_a + *ai * *ai;
                norm_b = norm_b + *bi * *bi;
            }

            if norm_a < F::epsilon() || norm_b < F::epsilon() {
                return Err(MetricsError::InvalidInput(
                    "Cannot compute cosine distance for zero vectors".to_string(),
                ));
            }

            let cosine_similarity = dot / (norm_a.sqrt() * norm_b.sqrt());

            // Cosine distance is 1 - cosine similarity
            Ok(F::one() - cosine_similarity)
        }
        _ => Err(MetricsError::InvalidInput(format!(
            "Unknown distance metric: {metric}"
        ))),
    }
}

/// Compute pairwise distances between all points in a matrix
///
/// # Arguments
///
/// * `x` - Data matrix, shape (n_samples, n_features)
/// * `metric` - Distance metric name
///
/// # Returns
///
/// * Distance matrix, shape (n_samples, n_samples)
pub(crate) fn pairwise_distances<F, S>(x: &ArrayBase<S, Ix2>, metric: &str) -> Result<Array2<F>>
where
    F: Float + NumCast + std::fmt::Debug,
    S: Data<Elem = F>,
{
    let (n_samples, _n_features) = x.dim();
    let mut distances = Array2::zeros((n_samples, n_samples));

    for i in 0..n_samples {
        for j in i..n_samples {
            let row_i = x.row(i).to_vec();
            let row_j = x.row(j).to_vec();

            let dist = calculate_distance(&row_i, &row_j, metric)?;

            distances[[i, j]] = dist;
            if i != j {
                distances[[j, i]] = dist; // Distance matrix is symmetric
            }
        }
    }

    Ok(distances)
}

/// Group data points by cluster labels
///
/// # Arguments
///
/// * `x` - Data matrix, shape (n_samples, n_features)
/// * `labels` - Cluster labels, shape (n_samples,)
///
/// # Returns
///
/// * HashMap mapping cluster labels to indices of points in that cluster
pub(crate) fn group_by_labels<F, T, S1, S2, D>(
    x: &ArrayBase<S1, Ix2>,
    labels: &ArrayBase<S2, D>,
) -> Result<HashMap<T, Vec<usize>>>
where
    F: Float + NumCast + std::fmt::Debug,
    T: std::hash::Hash + std::cmp::Eq + Copy,
    S1: Data<Elem = F>,
    S2: Data<Elem = T>,
    D: Dimension,
{
    let n_samples = x.shape()[0];

    if labels.len() != n_samples {
        return Err(MetricsError::InvalidInput(format!(
            "Number of labels ({}) does not match number of samples ({})",
            labels.len(),
            n_samples
        )));
    }

    let mut clusters: HashMap<T, Vec<usize>> = HashMap::new();

    for (i, &label) in labels.iter().enumerate() {
        clusters.entry(label).or_default().push(i);
    }

    Ok(clusters)
}
