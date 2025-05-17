//! Clustering evaluation metrics
//!
//! This module provides metrics for evaluating clustering algorithms performance:
//! - Silhouette coefficient for measuring cluster cohesion and separation
//! - Davies-Bouldin index for evaluating cluster separation
//! - Calinski-Harabasz index for measuring between-cluster vs within-cluster variance

mod silhouette;
pub use silhouette::{silhouette_samples, silhouette_score};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};

/// Compute the inertia (sum of squared distances) from samples to their cluster centers.
///
/// This is a utility function used by other metrics.
///
/// # Arguments
///
/// * `data` - Input data (n_samples x n_features)
/// * `labels` - Cluster labels for each sample
/// * `centers` - Cluster centers (n_clusters x n_features)
///
/// # Returns
///
/// The sum of squared distances from samples to their cluster centers.
#[allow(dead_code)]
fn compute_inertia<F>(data: ArrayView2<F>, labels: ArrayView1<i32>, centers: ArrayView2<F>) -> F
where
    F: Float + FromPrimitive + 'static,
{
    let mut inertia = F::zero();

    for (i, sample) in data.outer_iter().enumerate() {
        let label = labels[i];
        if label >= 0 && (label as usize) < centers.shape()[0] {
            let center = centers.row(label as usize);
            let diff = &sample - &center;
            let squared_distance = diff.dot(&diff);
            inertia = inertia + squared_distance;
        }
    }

    inertia
}

/// Davies-Bouldin score for clustering evaluation.
///
/// The Davies-Bouldin index measures the average similarity between clusters,
/// where similarity is the ratio of within-cluster and between-cluster distances.
/// A lower score indicates better clustering.
///
/// # Arguments
///
/// * `data` - Input data (n_samples x n_features)
/// * `labels` - Cluster labels for each sample
///
/// # Returns
///
/// The Davies-Bouldin score (lower is better)
///
/// # Example
///
/// ```
/// use ndarray::{Array1, Array2};
/// use scirs2_cluster::metrics::davies_bouldin_score;
///
/// let data = Array2::from_shape_vec((4, 2), vec![
///     0.0, 0.0,
///     0.1, 0.1,
///     5.0, 5.0,
///     5.1, 5.1,
/// ]).unwrap();
/// let labels = Array1::from_vec(vec![0, 0, 1, 1]);
///
/// let score = davies_bouldin_score(data.view(), labels.view()).unwrap();
/// assert!(score < 0.5);  // Should be low for well-separated clusters
/// ```
pub fn davies_bouldin_score<F>(data: ArrayView2<F>, labels: ArrayView1<i32>) -> Result<F>
where
    F: Float + FromPrimitive + Debug + PartialOrd + 'static,
{
    if data.shape()[0] != labels.shape()[0] {
        return Err(ClusteringError::InvalidInput(
            "Data and labels must have the same number of samples".to_string(),
        ));
    }

    // Find unique cluster labels
    let mut unique_labels = Vec::new();
    for &label in labels.iter() {
        if label >= 0 && !unique_labels.contains(&label) {
            unique_labels.push(label);
        }
    }

    let n_clusters = unique_labels.len();

    if n_clusters < 2 {
        return Err(ClusteringError::InvalidInput(
            "Davies-Bouldin score requires at least 2 clusters".to_string(),
        ));
    }

    // Compute cluster centers
    let mut centers = Array2::<F>::zeros((n_clusters, data.shape()[1]));
    let mut cluster_sizes = vec![0; n_clusters];

    for (i, &label) in labels.iter().enumerate() {
        if label >= 0 {
            let cluster_idx = unique_labels.iter().position(|&l| l == label).unwrap();
            centers
                .row_mut(cluster_idx)
                .scaled_add(F::one(), &data.row(i));
            cluster_sizes[cluster_idx] += 1;
        }
    }

    // Normalize to get averages
    for (i, &size) in cluster_sizes.iter().enumerate() {
        if size > 0 {
            centers
                .row_mut(i)
                .mapv_inplace(|x| x / F::from(size).unwrap());
        }
    }

    // Compute within-cluster scatter
    let mut scatter = vec![F::zero(); n_clusters];
    for (i, &label) in labels.iter().enumerate() {
        if label >= 0 {
            let cluster_idx = unique_labels.iter().position(|&l| l == label).unwrap();
            let center = centers.row(cluster_idx);
            let diff = &data.row(i) - &center;
            let distance = diff.dot(&diff).sqrt();
            scatter[cluster_idx] = scatter[cluster_idx] + distance;
        }
    }

    // Compute average within-cluster scatter
    for (i, &size) in cluster_sizes.iter().enumerate() {
        if size > 0 {
            scatter[i] = scatter[i] / F::from(size).unwrap();
        }
    }

    // Compute Davies-Bouldin index
    let mut db_index = F::zero();

    for i in 0..n_clusters {
        let mut max_ratio = F::zero();

        for j in 0..n_clusters {
            if i != j {
                let between_distance = (&centers.row(i) - &centers.row(j))
                    .mapv(|x| x * x)
                    .sum()
                    .sqrt();

                if between_distance > F::zero() {
                    let ratio = (scatter[i] + scatter[j]) / between_distance;
                    if ratio > max_ratio {
                        max_ratio = ratio;
                    }
                }
            }
        }

        db_index = db_index + max_ratio;
    }

    db_index = db_index / F::from(n_clusters).unwrap();
    Ok(db_index)
}

/// Calinski-Harabasz score for clustering evaluation.
///
/// Also known as the Variance Ratio Criterion, this score computes the ratio of
/// the sum of between-clusters dispersion to the within-cluster dispersion.
/// A higher score indicates better defined clusters.
///
/// # Arguments
///
/// * `data` - Input data (n_samples x n_features)
/// * `labels` - Cluster labels for each sample
///
/// # Returns
///
/// The Calinski-Harabasz score (higher is better)
///
/// # Example
///
/// ```
/// use ndarray::{Array1, Array2};
/// use scirs2_cluster::metrics::calinski_harabasz_score;
///
/// let data = Array2::from_shape_vec((4, 2), vec![
///     0.0, 0.0,
///     0.1, 0.1,
///     5.0, 5.0,
///     5.1, 5.1,
/// ]).unwrap();
/// let labels = Array1::from_vec(vec![0, 0, 1, 1]);
///
/// let score = calinski_harabasz_score(data.view(), labels.view()).unwrap();
/// assert!(score > 50.0);  // Should be high for well-separated clusters
/// ```
pub fn calinski_harabasz_score<F>(data: ArrayView2<F>, labels: ArrayView1<i32>) -> Result<F>
where
    F: Float + FromPrimitive + Debug + PartialOrd + 'static,
{
    if data.shape()[0] != labels.shape()[0] {
        return Err(ClusteringError::InvalidInput(
            "Data and labels must have the same number of samples".to_string(),
        ));
    }

    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];

    // Find unique cluster labels
    let mut unique_labels = Vec::new();
    for &label in labels.iter() {
        if label >= 0 && !unique_labels.contains(&label) {
            unique_labels.push(label);
        }
    }

    let n_clusters = unique_labels.len();

    if n_clusters < 2 {
        return Err(ClusteringError::InvalidInput(
            "Calinski-Harabasz score requires at least 2 clusters".to_string(),
        ));
    }

    if n_clusters >= n_samples {
        return Err(ClusteringError::InvalidInput(
            "Number of clusters must be less than number of samples".to_string(),
        ));
    }

    // Compute overall mean
    let mut overall_mean = Array1::<F>::zeros(n_features);
    let mut valid_samples = 0;

    for (i, &label) in labels.iter().enumerate() {
        if label >= 0 {
            overall_mean.scaled_add(F::one(), &data.row(i));
            valid_samples += 1;
        }
    }

    overall_mean.mapv_inplace(|x| x / F::from(valid_samples).unwrap());

    // Compute cluster centers and sizes
    let mut centers = Array2::<F>::zeros((n_clusters, n_features));
    let mut cluster_sizes = vec![0; n_clusters];

    for (i, &label) in labels.iter().enumerate() {
        if label >= 0 {
            let cluster_idx = unique_labels.iter().position(|&l| l == label).unwrap();
            centers
                .row_mut(cluster_idx)
                .scaled_add(F::one(), &data.row(i));
            cluster_sizes[cluster_idx] += 1;
        }
    }

    // Normalize to get averages
    for (i, &size) in cluster_sizes.iter().enumerate() {
        if size > 0 {
            centers
                .row_mut(i)
                .mapv_inplace(|x| x / F::from(size).unwrap());
        }
    }

    // Compute between-group sum of squares (SSB)
    let mut ssb = F::zero();
    for (i, &size) in cluster_sizes.iter().enumerate() {
        if size > 0 {
            let diff = &centers.row(i) - &overall_mean;
            ssb = ssb + F::from(size).unwrap() * diff.dot(&diff);
        }
    }

    // Compute within-group sum of squares (SSW)
    let mut ssw = F::zero();
    for (i, &label) in labels.iter().enumerate() {
        if label >= 0 {
            let cluster_idx = unique_labels.iter().position(|&l| l == label).unwrap();
            let diff = &data.row(i) - &centers.row(cluster_idx);
            ssw = ssw + diff.dot(&diff);
        }
    }

    // Calculate score
    if ssw == F::zero() {
        return Ok(F::infinity());
    }

    let score = (ssb / ssw) * F::from(valid_samples - n_clusters).unwrap()
        / F::from(n_clusters - 1).unwrap();

    Ok(score)
}

/// Mean silhouette coefficient over all samples.
///
/// Convenience wrapper that computes the mean silhouette coefficient using
/// Euclidean distance metric by default.
///
/// # Example
///
/// ```
/// use ndarray::{Array1, Array2};
/// use scirs2_cluster::metrics::mean_silhouette_score;
///
/// let data = Array2::from_shape_vec((4, 2), vec![
///     0.0, 0.0,
///     0.1, 0.1,
///     5.0, 5.0,
///     5.1, 5.1,
/// ]).unwrap();
/// let labels = Array1::from_vec(vec![0, 0, 1, 1]);
///
/// let score = mean_silhouette_score(data.view(), labels.view()).unwrap();
/// ```
pub fn mean_silhouette_score<F>(data: ArrayView2<F>, labels: ArrayView1<i32>) -> Result<F>
where
    F: Float + FromPrimitive + 'static,
{
    silhouette_score(data, labels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    #[test]
    fn test_davies_bouldin_score() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 5.0, 5.0, 5.1, 5.1, 5.2, 5.2],
        )
        .unwrap();
        let labels = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);

        let score = davies_bouldin_score(data.view(), labels.view()).unwrap();
        assert!(score < 0.5); // Should be low for well-separated clusters
    }

    #[test]
    fn test_calinski_harabasz_score() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 5.0, 5.0, 5.1, 5.1, 5.2, 5.2],
        )
        .unwrap();
        let labels = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);

        let score = calinski_harabasz_score(data.view(), labels.view()).unwrap();
        assert!(score > 50.0); // Should be high for well-separated clusters
    }
}
