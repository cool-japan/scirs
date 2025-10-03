//! Cluster distance metrics module
//!
//! This module provides metrics for measuring distances between clusters
//! and within clusters. These metrics are useful for evaluating the
//! quality of clustering results without external ground truth.

use scirs2_core::ndarray::{Array1, ArrayBase, Data, Dimension, Ix1, Ix2};
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::collections::{HashMap, HashSet};
use std::ops::{AddAssign, DivAssign};

use crate::error::{MetricsError, Result};

/// Calculate inter-cluster distances between all pairs of clusters
///
/// Inter-cluster distance measures the separation between different clusters.
/// Higher values indicate better separation between clusters.
///
/// # Arguments
///
/// * `x` - Array of shape (n_samples, n_features) - The data
/// * `labels` - Array of shape (n_samples,) - Predicted labels for each sample
/// * `metric` - Distance metric to use ("euclidean", "manhattan", or "cosine")
///
/// # Returns
///
/// * A HashMap mapping each cluster pair (i, j) to their distance
///
/// # Examples
///
/// ```no_run
/// use scirs2_core::ndarray::{array, Array2};
/// use scirs2_metrics::clustering::distance::inter_cluster_distances;
///
/// // Create a small dataset with 2 clusters
/// let x = Array2::<f64>::from_shape_vec((6, 2), vec![
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
/// let distances: std::collections::HashMap<(usize, usize), f64> = inter_cluster_distances(&x, &labels, "euclidean").unwrap();
/// ```
#[allow(dead_code)]
pub fn inter_cluster_distances<F, S1, S2, D>(
    x: &ArrayBase<S1, Ix2>,
    labels: &ArrayBase<S2, D>,
    metric: &str,
) -> Result<HashMap<(usize, usize), F>>
where
    F: Float
        + NumCast
        + std::fmt::Debug
        + scirs2_core::ndarray::ScalarOperand
        + AddAssign
        + DivAssign
        + SimdUnifiedOps,
    S1: Data<Elem = F>,
    S2: Data<Elem = usize>,
    D: Dimension,
{
    // Check that the metric is supported
    if !["euclidean", "manhattan", "cosine"].contains(&metric) {
        return Err(MetricsError::InvalidInput(format!(
            "Unsupported metric: {metric}. Supported metrics are 'euclidean', 'manhattan', and 'cosine'."
        )));
    }

    // Check that x and labels have the same number of samples
    let n_samples = x.shape()[0];
    if n_samples != labels.len() {
        return Err(MetricsError::InvalidInput(format!(
            "x has {} samples, but labels has {} samples",
            n_samples,
            labels.len()
        )));
    }

    // Get unique cluster labels efficiently
    let unique_set: HashSet<usize> = labels.iter().copied().collect();
    let mut unique_labels: Vec<usize> = unique_set.into_iter().collect();

    // Sort labels for consistent results
    unique_labels.sort();

    // Calculate cluster centroids
    let n_features = x.shape()[1];
    let mut centroids = HashMap::new();

    for &label in &unique_labels {
        let mut centroid = Array1::zeros(n_features);
        let mut count = 0;

        for (i, &sample_label) in labels.iter().enumerate() {
            if sample_label == label {
                let sample = x.slice(scirs2_core::ndarray::s![i, ..]);
                centroid += &sample;
                count += 1;
            }
        }

        if count > 0 {
            centroid /= F::from(count).unwrap();
            centroids.insert(label, centroid);
        }
    }

    // Calculate distances between all pairs of centroids
    let mut distances = HashMap::new();

    for (i, &label_i) in unique_labels.iter().enumerate() {
        for &label_j in unique_labels.iter().skip(i + 1) {
            let centroid_i = centroids.get(&label_i).unwrap();
            let centroid_j = centroids.get(&label_j).unwrap();

            let distance = match metric {
                "euclidean" => euclidean_distance(centroid_i, centroid_j),
                "manhattan" => manhattan_distance(centroid_i, centroid_j),
                "cosine" => cosine_distance(centroid_i, centroid_j),
                _ => {
                    return Err(MetricsError::InvalidInput(format!(
                        "Unsupported metric: {metric}"
                    )))
                }
            };

            distances.insert((label_i, label_j), distance);
            distances.insert((label_j, label_i), distance); // Symmetric
        }
    }

    Ok(distances)
}

/// Calculate intra-cluster distances for all clusters
///
/// Intra-cluster distance measures the cohesion within a cluster.
/// Lower values indicate more compact clusters.
///
/// # Arguments
///
/// * `x` - Array of shape (n_samples, n_features) - The data
/// * `labels` - Array of shape (n_samples,) - Predicted labels for each sample
/// * `metric` - Distance metric to use ("euclidean", "manhattan", or "cosine")
///
/// # Returns
///
/// * A HashMap mapping each cluster label to its intra-cluster distance
///
/// # Examples
///
/// ```no_run
/// use scirs2_core::ndarray::{array, Array2};
/// use scirs2_metrics::clustering::distance::intra_cluster_distances;
///
/// // Create a small dataset with 2 clusters
/// let x = Array2::<f64>::from_shape_vec((6, 2), vec![
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
/// let distances: std::collections::HashMap<usize, f64> = intra_cluster_distances(&x, &labels, "euclidean").unwrap();
/// ```
#[allow(dead_code)]
pub fn intra_cluster_distances<F, S1, S2, D>(
    x: &ArrayBase<S1, Ix2>,
    labels: &ArrayBase<S2, D>,
    metric: &str,
) -> Result<HashMap<usize, F>>
where
    F: Float
        + NumCast
        + std::fmt::Debug
        + scirs2_core::ndarray::ScalarOperand
        + AddAssign
        + DivAssign
        + SimdUnifiedOps,
    S1: Data<Elem = F>,
    S2: Data<Elem = usize>,
    D: Dimension,
{
    // Check that the metric is supported
    if !["euclidean", "manhattan", "cosine"].contains(&metric) {
        return Err(MetricsError::InvalidInput(format!(
            "Unsupported metric: {metric}. Supported metrics are 'euclidean', 'manhattan', and 'cosine'."
        )));
    }

    // Check that x and labels have the same number of samples
    let n_samples = x.shape()[0];
    if n_samples != labels.len() {
        return Err(MetricsError::InvalidInput(format!(
            "x has {} samples, but labels has {} samples",
            n_samples,
            labels.len()
        )));
    }

    // Get unique cluster labels efficiently
    let unique_set: HashSet<usize> = labels.iter().copied().collect();
    let mut unique_labels: Vec<usize> = unique_set.into_iter().collect();

    // Sort labels for consistent results
    unique_labels.sort();

    // Calculate cluster centroids
    let n_features = x.shape()[1];
    let mut centroids = HashMap::new();

    for &label in &unique_labels {
        let mut centroid = Array1::zeros(n_features);
        let mut count = 0;

        for (i, &sample_label) in labels.iter().enumerate() {
            if sample_label == label {
                let sample = x.slice(scirs2_core::ndarray::s![i, ..]);
                centroid += &sample;
                count += 1;
            }
        }

        if count > 0 {
            centroid /= F::from(count).unwrap();
            centroids.insert(label, centroid);
        }
    }

    // Calculate average distance from each point to its cluster centroid
    let mut distances = HashMap::new();

    for &label in &unique_labels {
        let centroid = centroids.get(&label).unwrap();
        let mut total_distance = F::zero();
        let mut count = 0;

        for (i, &sample_label) in labels.iter().enumerate() {
            if sample_label == label {
                let sample = x.slice(scirs2_core::ndarray::s![i, ..]);

                let distance = match metric {
                    "euclidean" => euclidean_distance(&sample, centroid),
                    "manhattan" => manhattan_distance(&sample, centroid),
                    "cosine" => cosine_distance(&sample, centroid),
                    _ => {
                        return Err(MetricsError::InvalidInput(format!(
                            "Unsupported metric: {metric}"
                        )))
                    }
                };

                total_distance += distance;
                count += 1;
            }
        }

        if count > 0 {
            let avg_distance = total_distance / F::from(count).unwrap();
            distances.insert(label, avg_distance);
        }
    }

    Ok(distances)
}

/// Calculate distance ratio index (Davies-Bouldin Index derivative)
///
/// The distance ratio index is a measure of cluster quality based on the
/// ratio of within-cluster distance to between-cluster distance.
/// Lower values indicate better clustering.
///
/// # Arguments
///
/// * `x` - Array of shape (n_samples, n_features) - The data
/// * `labels` - Array of shape (n_samples,) - Predicted labels for each sample
/// * `metric` - Distance metric to use ("euclidean", "manhattan", or "cosine")
///
/// # Returns
///
/// * The distance ratio index
///
/// # Examples
///
/// ```no_run
/// use scirs2_core::ndarray::{array, Array2};
/// use scirs2_metrics::clustering::distance::distance_ratio_index;
///
/// // Create a small dataset with 2 clusters
/// let x = Array2::<f64>::from_shape_vec((6, 2), vec![
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
/// let index: f64 = distance_ratio_index(&x, &labels, "euclidean").unwrap();
/// ```
#[allow(dead_code)]
pub fn distance_ratio_index<F, S1, S2, D>(
    x: &ArrayBase<S1, Ix2>,
    labels: &ArrayBase<S2, D>,
    metric: &str,
) -> Result<F>
where
    F: Float
        + NumCast
        + std::fmt::Debug
        + scirs2_core::ndarray::ScalarOperand
        + AddAssign
        + DivAssign
        + SimdUnifiedOps,
    S1: Data<Elem = F>,
    S2: Data<Elem = usize>,
    D: Dimension,
{
    // Get inter-cluster and intra-cluster distances
    let inter_distances = inter_cluster_distances(x, labels, metric)?;
    let intra_distances = intra_cluster_distances(x, labels, metric)?;

    // Get unique cluster labels efficiently
    let unique_set: HashSet<usize> = labels.iter().copied().collect();
    let mut unique_labels: Vec<usize> = unique_set.into_iter().collect();

    // Sort labels for consistent results
    unique_labels.sort();

    // If there's only one cluster, return infinity
    if unique_labels.len() <= 1 {
        return Ok(F::infinity());
    }

    // Calculate the Davies-Bouldin-like ratio for each cluster
    let mut cluster_ratios = Vec::new();

    for (i, &label_i) in unique_labels.iter().enumerate() {
        let mut max_ratio = F::zero();

        for &label_j in unique_labels.iter().skip(i + 1) {
            // Skip comparison with self
            if label_i == label_j {
                continue;
            }

            let intra_i = *intra_distances.get(&label_i).unwrap_or(&F::zero());
            let intra_j = *intra_distances.get(&label_j).unwrap_or(&F::zero());
            let inter_ij = *inter_distances
                .get(&(label_i, label_j))
                .unwrap_or(&F::infinity());

            // Calculate the ratio (sum of intra-cluster distances / inter-cluster distance)
            let ratio = (intra_i + intra_j) / inter_ij;

            if ratio > max_ratio {
                max_ratio = ratio;
            }
        }

        cluster_ratios.push(max_ratio);
    }

    // Calculate the average ratio across all clusters
    let sum_ratios = cluster_ratios.iter().fold(F::zero(), |acc, &x| acc + x);
    let avg_ratio = sum_ratios / F::from(cluster_ratios.len()).unwrap();

    Ok(avg_ratio)
}

/// Calculate isolation index
///
/// The isolation index measures how well-separated the clusters are
/// in comparison to their internal cohesion. Higher values indicate
/// better separation between clusters.
///
/// # Arguments
///
/// * `x` - Array of shape (n_samples, n_features) - The data
/// * `labels` - Array of shape (n_samples,) - Predicted labels for each sample
/// * `metric` - Distance metric to use ("euclidean", "manhattan", or "cosine")
///
/// # Returns
///
/// * The isolation index
///
/// # Examples
///
/// ```no_run
/// use scirs2_core::ndarray::{array, Array2};
/// use scirs2_metrics::clustering::distance::isolation_index;
///
/// // Create a small dataset with 2 clusters
/// let x = Array2::<f64>::from_shape_vec((6, 2), vec![
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
/// let index: f64 = isolation_index(&x, &labels, "euclidean").unwrap();
/// ```
#[allow(dead_code)]
pub fn isolation_index<F, S1, S2, D>(
    x: &ArrayBase<S1, Ix2>,
    labels: &ArrayBase<S2, D>,
    metric: &str,
) -> Result<F>
where
    F: Float
        + NumCast
        + std::fmt::Debug
        + scirs2_core::ndarray::ScalarOperand
        + AddAssign
        + DivAssign
        + SimdUnifiedOps,
    S1: Data<Elem = F>,
    S2: Data<Elem = usize>,
    D: Dimension,
{
    // Get inter-cluster and intra-cluster distances
    let inter_distances = inter_cluster_distances(x, labels, metric)?;
    let intra_distances = intra_cluster_distances(x, labels, metric)?;

    // Get unique cluster labels efficiently
    let unique_set: HashSet<usize> = labels.iter().copied().collect();
    let mut unique_labels: Vec<usize> = unique_set.into_iter().collect();

    // Sort labels for consistent results
    unique_labels.sort();

    // If there's only one cluster, return zero
    if unique_labels.len() <= 1 {
        return Ok(F::zero());
    }

    // Calculate the minimum inter-cluster distance
    let min_inter =
        inter_distances.values().fold(
            F::infinity(),
            |min_dist, &dist| {
                if dist < min_dist {
                    dist
                } else {
                    min_dist
                }
            },
        );

    // Calculate the maximum intra-cluster distance
    let max_intra =
        intra_distances.values().fold(
            F::zero(),
            |max_dist, &dist| {
                if dist > max_dist {
                    dist
                } else {
                    max_dist
                }
            },
        );

    // Calculate the isolation index (min inter-cluster / max intra-cluster)
    // Higher values are better (indicating good separation and cohesion)
    let isolation = if max_intra > F::zero() {
        min_inter / max_intra
    } else {
        F::infinity() // If all intra-cluster distances are zero, clusters are perfectly cohesive
    };

    Ok(isolation)
}

// Distance metric functions

#[allow(dead_code)]
fn euclidean_distance<F, S1, S2>(x: &ArrayBase<S1, Ix1>, y: &ArrayBase<S2, Ix1>) -> F
where
    F: Float + scirs2_core::ndarray::ScalarOperand + SimdUnifiedOps,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
{
    // Use SIMD optimizations for contiguous arrays
    if x.is_standard_layout() && y.is_standard_layout() {
        let diff = F::simd_sub(&x.view(), &y.view());
        let squared_diff = F::simd_mul(&diff.view(), &diff.view());
        F::simd_sum(&squared_diff.view()).sqrt()
    } else {
        // Fallback for non-contiguous arrays
        let mut sum_sq = F::zero();
        for (a, b) in x.iter().zip(y.iter()) {
            let diff = *a - *b;
            sum_sq = sum_sq + diff * diff;
        }
        sum_sq.sqrt()
    }
}

#[allow(dead_code)]
fn manhattan_distance<F, S1, S2>(x: &ArrayBase<S1, Ix1>, y: &ArrayBase<S2, Ix1>) -> F
where
    F: Float + scirs2_core::ndarray::ScalarOperand + SimdUnifiedOps,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
{
    // Use SIMD optimizations for contiguous arrays
    if x.is_standard_layout() && y.is_standard_layout() {
        let diff = F::simd_sub(&x.view(), &y.view());
        let abs_diff = F::simd_abs(&diff.view());
        F::simd_sum(&abs_diff.view())
    } else {
        // Fallback for non-contiguous arrays
        let mut sum_abs = F::zero();
        for (a, b) in x.iter().zip(y.iter()) {
            let diff = *a - *b;
            sum_abs = sum_abs + diff.abs();
        }
        sum_abs
    }
}

#[allow(dead_code)]
fn cosine_distance<F, S1, S2>(x: &ArrayBase<S1, Ix1>, y: &ArrayBase<S2, Ix1>) -> F
where
    F: Float + scirs2_core::ndarray::ScalarOperand + SimdUnifiedOps,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
{
    // Use SIMD optimizations for contiguous arrays
    let (dot_product, norm_x, norm_y) = if x.is_standard_layout() && y.is_standard_layout() {
        let xy = F::simd_mul(&x.view(), &y.view());
        let dot_product = F::simd_sum(&xy.view());

        let x_squared = F::simd_mul(&x.view(), &x.view());
        let norm_x_sq = F::simd_sum(&x_squared.view());

        let y_squared = F::simd_mul(&y.view(), &y.view());
        let norm_y_sq = F::simd_sum(&y_squared.view());

        (dot_product, norm_x_sq.sqrt(), norm_y_sq.sqrt())
    } else {
        // Fallback for non-contiguous arrays
        let mut dot_product = F::zero();
        let mut norm_x = F::zero();
        let mut norm_y = F::zero();

        for (a, b) in x.iter().zip(y.iter()) {
            dot_product = dot_product + (*a * *b);
            norm_x = norm_x + (*a * *a);
            norm_y = norm_y + (*b * *b);
        }

        (dot_product, norm_x.sqrt(), norm_y.sqrt())
    };

    if norm_x > F::zero() && norm_y > F::zero() {
        F::one() - (dot_product / (norm_x * norm_y))
    } else {
        F::one() // If either vector is zero, use maximum distance
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    #[ignore = "timeout"]
    fn test_inter_cluster_distances_euclidean() {
        // Create a simple dataset with 2 clearly separated clusters
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.5, 1.8, 1.2, 2.2, 5.0, 6.0, 5.2, 5.8, 5.5, 6.2],
        )
        .unwrap();

        let labels = array![0, 0, 0, 1, 1, 1];

        let distances = inter_cluster_distances(&x, &labels, "euclidean").unwrap();

        // Check that the distance between clusters 0 and 1 is reasonable
        let dist_0_1 = distances.get(&(0, 1)).unwrap();
        assert!(*dist_0_1 > 4.0); // Clusters should be well-separated

        // Distance should be symmetric
        let dist_1_0 = distances.get(&(1, 0)).unwrap();
        assert_abs_diff_eq!(*dist_0_1, *dist_1_0, epsilon = 1e-10);
    }

    #[test]
    #[ignore = "timeout"]
    fn test_intra_cluster_distances_euclidean() {
        // Create a simple dataset with 2 compact clusters
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.5, 1.8, 1.2, 2.2, 5.0, 6.0, 5.2, 5.8, 5.5, 6.2],
        )
        .unwrap();

        let labels = array![0, 0, 0, 1, 1, 1];

        let distances = intra_cluster_distances(&x, &labels, "euclidean").unwrap();

        // Check that the intra-cluster distances are reasonable
        let dist_0 = distances.get(&0).unwrap();
        let dist_1 = distances.get(&1).unwrap();

        // Each cluster should be compact (small intra-cluster distance)
        assert!(*dist_0 < 1.0);
        assert!(*dist_1 < 1.0);
    }

    #[test]
    #[ignore = "timeout"]
    fn test_distance_ratio_index() {
        // Create a dataset with well-separated clusters
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.5, 1.8, 1.2, 2.2, 5.0, 6.0, 5.2, 5.8, 5.5, 6.2],
        )
        .unwrap();

        let labels = array![0, 0, 0, 1, 1, 1];

        let index = distance_ratio_index(&x, &labels, "euclidean").unwrap();

        // Well-separated clusters should have a low ratio index
        assert!(index < 0.5);
    }

    #[test]
    #[ignore = "timeout"]
    fn test_isolation_index() {
        // Create a dataset with well-separated clusters
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.5, 1.8, 1.2, 2.2, 5.0, 6.0, 5.2, 5.8, 5.5, 6.2],
        )
        .unwrap();

        let labels = array![0, 0, 0, 1, 1, 1];

        let index = isolation_index(&x, &labels, "euclidean").unwrap();

        // Well-separated clusters should have a high isolation index
        assert!(index > 2.0);
    }
}
