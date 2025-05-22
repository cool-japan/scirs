//! Clustering evaluation utilities
//!
//! This module provides utilities for evaluating clustering results, including
//! metrics like Dunn index and elbow method for determining optimal number of clusters.

use ndarray::{Array1, Array2, ArrayBase, Data, Ix1, Ix2};
use num_traits::{Float, NumCast};
use std::collections::HashMap;

use super::{calculate_distance, group_by_labels, pairwise_distances};
use crate::error::{MetricsError, Result};

/// Calculates the enhanced Dunn index for a clustering (evaluation version)
///
/// The Dunn index is defined as the ratio of the smallest distance between clusters
/// to the largest intra-cluster distance (cluster diameter). Higher values indicate
/// better clustering with dense, well-separated clusters.
///
/// This is an alternative implementation for the evaluation module.
///
/// # Arguments
///
/// * `x` - Array of shape (n_samples, n_features) - The data
/// * `labels` - Array of shape (n_samples,) - Predicted labels for each sample
///
/// # Returns
///
/// * The Dunn index
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array2};
/// // This function is not re-exported at the top level
/// use scirs2_metrics::clustering::evaluation::dunn_index_enhanced;
///
/// // Create a small dataset with 2 clusters
/// let x = Array2::from_shape_vec((6, 2), vec![
///     1.0, 2.0,
///     1.5, 1.8,
///     1.2, 2.2,
///     5.0, 6.0,
///     5.2, 5.8,
///     5.5, 6.2,
/// ]).unwrap();
///
/// let labels = array![0, 0, 0, 1, 1, 1];
///
/// let score = dunn_index_enhanced(&x, &labels).unwrap();
/// assert!(score > 0.5); // High score for well-separated clusters
/// ```
pub fn dunn_index_enhanced<F, S1, S2, D>(
    x: &ArrayBase<S1, Ix2>,
    labels: &ArrayBase<S2, D>,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug,
    S1: Data<Elem = F>,
    S2: Data<Elem = usize>,
    D: ndarray::Dimension,
{
    // Check dimensions
    let n_samples = x.shape()[0];
    if labels.len() != n_samples {
        return Err(MetricsError::InvalidInput(format!(
            "Number of samples in x ({}) does not match number of labels ({})",
            n_samples,
            labels.len()
        )));
    }

    // Group data points by cluster
    let clusters = group_by_labels(x, labels)?;
    let n_clusters = clusters.len();

    if n_clusters <= 1 {
        return Err(MetricsError::InvalidInput(
            "Dunn index is only defined for more than one cluster".to_string(),
        ));
    }

    // Calculate pairwise distances within the dataset
    let distances = pairwise_distances::<F, _>(x, "euclidean")?;

    // Find minimum inter-cluster distance
    let mut min_inter_distance = F::infinity();

    for (i, indices_i) in clusters.iter() {
        for (j, indices_j) in clusters.iter() {
            if i != j {
                // Calculate minimum distance between points in different clusters
                for &idx_i in indices_i {
                    for &idx_j in indices_j {
                        min_inter_distance = min_inter_distance.min(distances[[idx_i, idx_j]]);
                    }
                }
            }
        }
    }

    // Find maximum intra-cluster distance (cluster diameter)
    let mut max_intra_distance = F::zero();

    for (_, indices) in clusters.iter() {
        for (idx, &i) in indices.iter().enumerate() {
            for &j in indices.iter().skip(idx + 1) {
                max_intra_distance = max_intra_distance.max(distances[[i, j]]);
            }
        }
    }

    // Calculate Dunn index
    if max_intra_distance == F::zero() {
        // Avoid division by zero
        Ok(F::infinity())
    } else {
        Ok(min_inter_distance / max_intra_distance)
    }
}

/// Implements the elbow method to determine the optimal number of clusters
///
/// The elbow method computes the sum of squared distances for a range of k values
/// and helps identify where the rate of decrease sharply changes (the "elbow").
///
/// # Arguments
///
/// * `x` - Data matrix, shape (n_samples, n_features)
/// * `k_range` - Range of k values to evaluate
/// * `kmeans_fn` - Function that runs k-means clustering and returns (centroids, labels, inertia)
///
/// # Returns
///
/// * Vector of inertia values for each k in k_range
///
/// # Examples
///
/// ```no_run
/// use ndarray::{Array1, Array2};
/// use scirs2_metrics::clustering::elbow_method;
///
/// // Create a dataset
/// let x = Array2::<f64>::zeros((100, 2));
///
/// // Define a function that runs k-means
/// let kmeans_fn = |data: &Array2<f64>, k: usize| {
///     // In a real example, you would call your actual k-means implementation
///     // This is just a placeholder
///     let inertia = k as f64; // Placeholder value
///     inertia
/// };
///
/// // Run elbow method for k from 1 to 10
/// let inertias = elbow_method(&x, 1..=10, kmeans_fn).unwrap();
///
/// // Now you can plot inertias against k to find the "elbow"
/// ```
pub fn elbow_method<F, S>(
    x: &ArrayBase<S, Ix2>,
    k_range: std::ops::RangeInclusive<usize>,
    kmeans_fn: impl Fn(&ArrayBase<S, Ix2>, usize) -> F,
) -> Result<Vec<F>>
where
    F: Float + NumCast + std::fmt::Debug,
    S: Data<Elem = F>,
{
    // Check if k_range is valid
    let start = *k_range.start();
    let end = *k_range.end();

    if start < 1 {
        return Err(MetricsError::InvalidInput(
            "k_range must start at 1 or greater".to_string(),
        ));
    }

    if end < start {
        return Err(MetricsError::InvalidInput(
            "k_range end must be greater than or equal to start".to_string(),
        ));
    }

    // Check data dimensions
    let (n_samples, n_features) = x.dim();
    if n_samples == 0 || n_features == 0 {
        return Err(MetricsError::InvalidInput(
            "Input data is empty".to_string(),
        ));
    }

    // Run k-means for each value of k and collect inertias
    let mut inertias = Vec::with_capacity(end - start + 1);
    for k in k_range {
        let inertia = kmeans_fn(x, k);
        inertias.push(inertia);
    }

    Ok(inertias)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::dunn_index;
    use ndarray::{array, Array2};

    #[test]
    fn test_dunn_index() {
        // Create a dataset with 2 well-separated clusters
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.5, 1.8, 1.2, 2.2, 5.0, 6.0, 5.2, 5.8, 5.5, 6.2],
        )
        .unwrap();

        let labels = array![0, 0, 0, 1, 1, 1];

        let dunn = dunn_index(&x, &labels).unwrap();
        // The clusters are well separated, so we expect a high Dunn index
        assert!(dunn > 0.5);

        // Create a dataset with overlapping clusters
        let x_overlap = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 2.0, 1.5, 1.8, 3.2, 3.2, // Overlap point
                3.0, 3.0, // Overlap point
                5.2, 5.8, 5.5, 6.2,
            ],
        )
        .unwrap();

        let labels_overlap = array![0, 0, 0, 1, 1, 1];

        let dunn_overlap = dunn_index(&x_overlap, &labels_overlap).unwrap();
        // Clusters are less separated, so we expect a lower Dunn index
        assert!(dunn_overlap < dunn);
    }

    #[test]
    fn test_elbow_method() {
        // Create a small dataset
        let x = Array2::<f64>::zeros((10, 2));

        // Simple mock k-means function that returns decreasing inertia values
        let kmeans_mock = |_: &Array2<f64>, k: usize| {
            // Simulate decreasing inertia with increasing k
            // with an "elbow" around k=3
            let base = 100.0;
            match k {
                1 => base,
                2 => base / 2.0,
                3 => base / 3.0, // Elbow point
                4 => base / 3.2, // Small decrease after elbow
                5 => base / 3.4,
                _ => base / (3.5 + (k as f64 - 5.0) * 0.1),
            }
        };

        let inertias = elbow_method(&x, 1..=6, kmeans_mock).unwrap();

        // Check expected properties
        assert_eq!(inertias.len(), 6);
        for i in 1..inertias.len() {
            // Inertia should always decrease
            assert!(inertias[i] < inertias[i - 1]);
        }

        // Check for elbow - the largest drop should be between k=1 and k=2,
        // with a smaller drop between k=2 and k=3, and minimal changes after
        let drop_1_to_2 = inertias[0] - inertias[1];
        let drop_2_to_3 = inertias[1] - inertias[2];
        let drop_3_to_4 = inertias[2] - inertias[3];

        assert!(drop_1_to_2 > drop_2_to_3);
        assert!(drop_2_to_3 > drop_3_to_4);
    }
}
