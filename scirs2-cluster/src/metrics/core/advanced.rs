//! Advanced clustering evaluation metrics
//!
//! This module provides advanced metrics for evaluating clustering quality,
//! including the Dunn index, Bayesian Information Criterion (BIC), and other
//! sophisticated clustering validation measures.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};

/// Compute the Dunn index for clustering evaluation.
///
/// The Dunn index is the ratio of the minimum inter-cluster distance to the maximum
/// intra-cluster distance. Higher values indicate better clustering with well-separated
/// and compact clusters.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `labels` - Cluster assignments for each sample
///
/// # Returns
/// * `Result<F>` - The Dunn index value
///
/// # Example
/// ```rust
/// use scirs2_core::ndarray::{Array1, Array2};
/// use scirs2_cluster::metrics::dunn_index;
///
/// let data = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 1.0, 10.0, 10.0, 11.0, 11.0]).unwrap();
/// let labels = Array1::from_vec(vec![0, 0, 1, 1]);
/// let dunn = dunn_index(data.view(), labels.view()).unwrap();
/// ```
pub fn dunn_index<F>(data: ArrayView2<F>, labels: ArrayView1<i32>) -> Result<F>
where
    F: Float + FromPrimitive + Debug + PartialOrd + 'static,
{
    if data.shape()[0] != labels.shape()[0] {
        return Err(ClusteringError::InvalidInput(
            "Data and labels must have the same number of samples".to_string(),
        ));
    }

    let n_samples = data.shape()[0];
    if n_samples < 2 {
        return Ok(F::zero());
    }

    // Find unique cluster labels
    let mut unique_labels = Vec::new();
    for &label in labels.iter() {
        if label >= 0 && !unique_labels.contains(&label) {
            unique_labels.push(label);
        }
    }

    if unique_labels.len() < 2 {
        return Ok(F::zero());
    }

    // Compute minimum inter-cluster distance
    let mut min_inter_cluster = F::infinity();
    for i in 0..n_samples {
        for j in (i + 1)..n_samples {
            if labels[i] >= 0 && labels[j] >= 0 && labels[i] != labels[j] {
                let distance = euclidean_distance(data.row(i), data.row(j));
                if distance < min_inter_cluster {
                    min_inter_cluster = distance;
                }
            }
        }
    }

    // Compute maximum intra-cluster distance
    let mut max_intra_cluster = F::zero();
    for &cluster_label in &unique_labels {
        let cluster_indices: Vec<usize> = labels
            .iter()
            .enumerate()
            .filter_map(|(i, &label)| {
                if label == cluster_label {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();

        for i in 0..cluster_indices.len() {
            for j in (i + 1)..cluster_indices.len() {
                let distance =
                    euclidean_distance(data.row(cluster_indices[i]), data.row(cluster_indices[j]));
                if distance > max_intra_cluster {
                    max_intra_cluster = distance;
                }
            }
        }
    }

    if max_intra_cluster == F::zero() {
        return Ok(F::infinity());
    }

    Ok(min_inter_cluster / max_intra_cluster)
}

/// Bayesian Information Criterion (BIC) for model selection.
///
/// Estimates the BIC for a clustering result, useful for determining
/// the optimal number of clusters. Lower BIC values indicate better models.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `labels` - Cluster assignments for each sample
///
/// # Returns
/// * `Result<F>` - The BIC score
///
/// # Example
/// ```rust
/// use scirs2_core::ndarray::{Array1, Array2};
/// use scirs2_cluster::metrics::bic_score;
///
/// let data = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 1.0, 10.0, 10.0, 11.0, 11.0]).unwrap();
/// let labels = Array1::from_vec(vec![0, 0, 1, 1]);
/// let bic = bic_score(data.view(), labels.view()).unwrap();
/// ```
pub fn bic_score<F>(data: ArrayView2<F>, labels: ArrayView1<i32>) -> Result<F>
where
    F: Float + FromPrimitive + Debug + PartialOrd + ScalarOperand + 'static,
{
    if data.shape()[0] != labels.shape()[0] {
        return Err(ClusteringError::InvalidInput(
            "Data and labels must have the same number of samples".to_string(),
        ));
    }

    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];

    if n_samples < 2 {
        return Ok(F::zero());
    }

    // Find unique cluster labels
    let mut unique_labels = Vec::new();
    for &label in labels.iter() {
        if label >= 0 && !unique_labels.contains(&label) {
            unique_labels.push(label);
        }
    }

    let k = unique_labels.len();
    if k < 1 {
        return Ok(F::infinity());
    }

    // Calculate within-cluster sum of squares
    let mut wcss = F::zero();
    for &cluster_label in &unique_labels {
        let cluster_indices: Vec<usize> = labels
            .iter()
            .enumerate()
            .filter_map(|(i, &label)| {
                if label == cluster_label {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();

        if cluster_indices.is_empty() {
            continue;
        }

        // Calculate cluster centroid
        let mut centroid = Array1::zeros(n_features);
        for &idx in &cluster_indices {
            centroid = &centroid + &data.row(idx);
        }
        centroid = centroid / F::from_usize(cluster_indices.len()).unwrap();

        // Calculate sum of squared distances to centroid
        for &idx in &cluster_indices {
            let diff = &data.row(idx) - &centroid;
            wcss = wcss + diff.dot(&diff);
        }
    }

    // Calculate BIC: log-likelihood + penalty term
    let n_params = k * (n_features + 1) - 1; // Number of parameters in the model
    let log_likelihood = -F::from_usize(n_samples).unwrap() * wcss.ln() / F::from_f64(2.0).unwrap();
    let penalty = F::from_usize(n_params).unwrap() * F::from_usize(n_samples).unwrap().ln()
        / F::from_f64(2.0).unwrap();

    Ok(-log_likelihood + penalty)
}

/// Akaike Information Criterion (AIC) for model selection.
///
/// Estimates the AIC for a clustering result, useful for determining
/// the optimal number of clusters. Lower AIC values indicate better models.
///
/// # Arguments
/// * `data` - Input data matrix (n_samples x n_features)
/// * `labels` - Cluster assignments for each sample
///
/// # Returns
/// * `Result<F>` - The AIC score
pub fn aic_score<F>(data: ArrayView2<F>, labels: ArrayView1<i32>) -> Result<F>
where
    F: Float + FromPrimitive + Debug + PartialOrd + ScalarOperand + 'static,
{
    if data.shape()[0] != labels.shape()[0] {
        return Err(ClusteringError::InvalidInput(
            "Data and labels must have the same number of samples".to_string(),
        ));
    }

    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];

    if n_samples < 2 {
        return Ok(F::zero());
    }

    // Find unique cluster labels
    let mut unique_labels = Vec::new();
    for &label in labels.iter() {
        if label >= 0 && !unique_labels.contains(&label) {
            unique_labels.push(label);
        }
    }

    let k = unique_labels.len();
    if k < 1 {
        return Ok(F::infinity());
    }

    // Calculate within-cluster sum of squares
    let mut wcss = F::zero();
    for &cluster_label in &unique_labels {
        let cluster_indices: Vec<usize> = labels
            .iter()
            .enumerate()
            .filter_map(|(i, &label)| {
                if label == cluster_label {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();

        if cluster_indices.is_empty() {
            continue;
        }

        // Calculate cluster centroid
        let mut centroid = Array1::zeros(n_features);
        for &idx in &cluster_indices {
            centroid = &centroid + &data.row(idx);
        }
        centroid = centroid / F::from_usize(cluster_indices.len()).unwrap();

        // Calculate sum of squared distances to centroid
        for &idx in &cluster_indices {
            let diff = &data.row(idx) - &centroid;
            wcss = wcss + diff.dot(&diff);
        }
    }

    // Calculate AIC: -2 * log-likelihood + 2 * k
    let n_params = k * (n_features + 1) - 1; // Number of parameters in the model
    let log_likelihood = -F::from_usize(n_samples).unwrap() * wcss.ln() / F::from_f64(2.0).unwrap();

    Ok(-F::from_f64(2.0).unwrap() * log_likelihood
        + F::from_f64(2.0).unwrap() * F::from_usize(n_params).unwrap())
}

/// Compute Euclidean distance between two points.
fn euclidean_distance<F>(point1: ArrayView1<F>, point2: ArrayView1<F>) -> F
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let diff = &point1 - &point2;
    diff.dot(&diff).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_dunn_index() {
        let data = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 1.0, 10.0, 10.0, 11.0, 11.0])
            .unwrap();
        let labels = Array1::from_vec(vec![0, 0, 1, 1]);

        let dunn = dunn_index(data.view(), labels.view()).unwrap();
        assert!(dunn > 0.0);
    }

    #[test]
    fn test_bic_score() {
        let data = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 1.0, 10.0, 10.0, 11.0, 11.0])
            .unwrap();
        let labels = Array1::from_vec(vec![0, 0, 1, 1]);

        let bic = bic_score(data.view(), labels.view()).unwrap();
        assert!(bic.is_finite());
    }

    #[test]
    fn test_aic_score() {
        let data = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 1.0, 10.0, 10.0, 11.0, 11.0])
            .unwrap();
        let labels = Array1::from_vec(vec![0, 0, 1, 1]);

        let aic = aic_score(data.view(), labels.view()).unwrap();
        assert!(aic.is_finite());
    }

    #[test]
    fn test_euclidean_distance() {
        let point1 = Array1::from_vec(vec![0.0, 0.0]);
        let point2 = Array1::from_vec(vec![3.0, 4.0]);

        let distance = euclidean_distance(point1.view(), point2.view());
        assert!((distance - 5.0).abs() < 1e-10);
    }
}
