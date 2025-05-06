//! Clustering evaluation metrics
//!
//! This module provides metrics for evaluating clustering algorithms performance:
//! - Silhouette coefficient for measuring cluster cohesion and separation
//! - Davies-Bouldin index for evaluating cluster separation
//! - Future: Calinski-Harabasz index, etc.

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};

/// Calculate the silhouette score (coefficient) for a clustering
///
/// The silhouette coefficient measures how well samples are clustered with
/// samples that are similar to themselves, compared to samples in other clusters.
/// The score ranges from -1 to 1, where:
/// * Values close to +1 indicate that samples are well-matched to their own clusters
///   and poorly matched to neighboring clusters
/// * Values close to 0 indicate overlapping clusters with samples on or near decision boundaries
/// * Negative values indicate samples may have been assigned to the wrong cluster
///
/// # Arguments
///
/// * `data` - The input data as a 2D array (n_samples x n_features)
/// * `labels` - The cluster labels for each sample (0 to k-1 for k clusters, -1 for noise)
/// * `metric` - Optional distance metric function: fn(&[F], &[F]) -> F
///
/// # Returns
///
/// The mean silhouette coefficient over all samples, or error if invalid input
///
/// # Examples
///
/// ```
/// use ndarray::{Array1, Array2};
/// use scirs2_cluster::metrics::silhouette_score;
///
/// // Example data with two clear clusters
/// let data = Array2::from_shape_vec((6, 2), vec![
///     1.0, 2.0,
///     1.2, 1.8,
///     0.8, 1.9,
///     4.0, 5.0,
///     4.2, 4.8,
///     3.9, 5.1,
/// ]).unwrap();
///
/// // Labels for the two clusters (0 and 1)
/// let labels = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);
///
/// // Calculate silhouette score
/// let score = silhouette_score(data.view(), labels.view(), None).unwrap();
/// println!("Silhouette score: {}", score);
///
/// // Should be close to 1.0 for well-separated clusters
/// assert!(score > 0.8);
/// ```
pub fn silhouette_score<F: Float + FromPrimitive + Debug + PartialOrd>(
    data: ArrayView2<F>,
    labels: ArrayView1<i32>,
    metric: Option<fn(&[F], &[F]) -> F>,
) -> Result<F> {
    // Check input dimensions
    if data.shape()[0] != labels.shape()[0] {
        return Err(ClusteringError::InvalidInput(
            "Number of samples in data and labels must match".into(),
        ));
    }

    let n_samples = data.shape()[0];
    if n_samples < 2 {
        return Err(ClusteringError::InvalidInput(
            "Silhouette score requires at least 2 samples".into(),
        ));
    }

    // Find the unique labels (excluding noise points with label -1)
    let mut unique_labels = Vec::new();
    for &label in labels.iter() {
        if label >= 0 && !unique_labels.contains(&label) {
            unique_labels.push(label);
        }
    }

    let n_clusters = unique_labels.len();
    if n_clusters < 2 || n_clusters >= n_samples {
        return Err(ClusteringError::InvalidInput(
            "Silhouette score requires at least 2 clusters and fewer clusters than samples".into(),
        ));
    }

    // Choose distance metric function
    let distance = metric.unwrap_or(|a, b| {
        // Default to Euclidean distance
        let mut sum = F::zero();
        for (a_i, b_i) in a.iter().zip(b.iter()) {
            let diff = *a_i - *b_i;
            sum = sum + diff * diff;
        }
        sum.sqrt()
    });

    // Calculate pairwise distances
    let mut distances = Array2::<F>::zeros((n_samples, n_samples));
    for i in 0..n_samples {
        for j in (i + 1)..n_samples {
            let dist = distance(&data.row(i).to_vec(), &data.row(j).to_vec());
            distances[[i, j]] = dist;
            distances[[j, i]] = dist; // Distance matrix is symmetric
        }
    }

    // For each sample, calculate silhouette value
    let mut silhouette_values = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let sample_label = labels[i];

        // Skip noise points (label -1)
        if sample_label < 0 {
            continue;
        }

        // Find all samples in the same cluster
        let mut same_cluster = Vec::new();
        for j in 0..n_samples {
            if i != j && labels[j] == sample_label {
                same_cluster.push(j);
            }
        }

        // If alone in cluster, silhouette is 0 (can't calculate intra-cluster distance)
        if same_cluster.is_empty() {
            silhouette_values.push(F::zero());
            continue;
        }

        // Calculate mean distance to all points in the same cluster (a)
        let mut intra_cluster_dist_sum = F::zero();
        for &j in &same_cluster {
            intra_cluster_dist_sum = intra_cluster_dist_sum + distances[[i, j]];
        }
        let a = intra_cluster_dist_sum / F::from_usize(same_cluster.len()).unwrap();

        // Find the mean distance to the closest neighboring cluster (b)
        let mut min_inter_cluster_dist = F::infinity();

        for &neighbor_label in &unique_labels {
            if neighbor_label == sample_label {
                continue;
            }

            // Calculate mean distance to this neighboring cluster
            let mut inter_cluster_dist_sum = F::zero();
            let mut count = 0;

            for j in 0..n_samples {
                if labels[j] == neighbor_label {
                    inter_cluster_dist_sum = inter_cluster_dist_sum + distances[[i, j]];
                    count += 1;
                }
            }

            if count > 0 {
                let mean_dist = inter_cluster_dist_sum / F::from_usize(count).unwrap();
                if mean_dist < min_inter_cluster_dist {
                    min_inter_cluster_dist = mean_dist;
                }
            }
        }

        let b = min_inter_cluster_dist;

        // Calculate silhouette value: (b - a) / max(a, b)
        let silhouette = (b - a) / a.max(b);
        silhouette_values.push(silhouette);
    }

    // Calculate mean silhouette value
    if silhouette_values.is_empty() {
        return Err(ClusteringError::ComputationError(
            "No valid samples for silhouette calculation".into(),
        ));
    }

    let sum = silhouette_values
        .iter()
        .fold(F::zero(), |acc, &val| acc + val);
    let mean = sum / F::from_usize(silhouette_values.len()).unwrap();

    Ok(mean)
}

/// Calculate the Davies-Bouldin index for a clustering
///
/// The Davies-Bouldin index is defined as the average similarity between each cluster
/// and its most similar cluster, where similarity is defined as the ratio of the sum
/// of within-cluster distances to the between-cluster distance.
/// The lower the value, the better the clustering.
///
/// # Arguments
///
/// * `data` - The input data as a 2D array (n_samples x n_features)
/// * `labels` - The cluster labels for each sample (0 to k-1 for k clusters, -1 for noise)
/// * `metric` - Optional distance metric function: fn(&[F], &[F]) -> F
///
/// # Returns
///
/// * `Result<F>` - The Davies-Bouldin index value, or error if invalid input
///
/// # Examples
///
/// ```
/// use ndarray::{Array1, Array2};
/// use scirs2_cluster::metrics::davies_bouldin_score;
///
/// // Example data with two clusters
/// let data = Array2::from_shape_vec((6, 2), vec![
///     1.0, 2.0,
///     1.2, 1.8,
///     0.8, 1.9,
///     5.0, 6.0,
///     5.2, 5.8,
///     4.8, 6.2,
/// ]).unwrap();
///
/// // Labels for the two clusters (0 and 1)
/// let labels = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);
///
/// // Calculate Davies-Bouldin score
/// let score = davies_bouldin_score(data.view(), labels.view(), None).unwrap();
/// println!("Davies-Bouldin score: {}", score);
///
/// // Should be a low value for well-separated clusters
/// assert!(score < 0.5);
/// ```
pub fn davies_bouldin_score<F: Float + FromPrimitive + Debug + PartialOrd>(
    data: ArrayView2<F>,
    labels: ArrayView1<i32>,
    metric: Option<fn(&[F], &[F]) -> F>,
) -> Result<F> {
    // Check input dimensions
    if data.shape()[0] != labels.shape()[0] {
        return Err(ClusteringError::InvalidInput(
            "Number of samples in data and labels must match".into(),
        ));
    }

    let n_samples = data.shape()[0];
    if n_samples < 2 {
        return Err(ClusteringError::InvalidInput(
            "Davies-Bouldin score requires at least 2 samples".into(),
        ));
    }

    // Find the unique labels (excluding noise points with label -1)
    let mut unique_labels = Vec::new();
    for &label in labels.iter() {
        if label >= 0 && !unique_labels.contains(&label) {
            unique_labels.push(label);
        }
    }

    let n_clusters = unique_labels.len();
    if n_clusters < 2 {
        return Err(ClusteringError::InvalidInput(
            "Davies-Bouldin score requires at least 2 clusters".into(),
        ));
    }

    // Choose distance metric function
    let distance = metric.unwrap_or(|a, b| {
        // Default to Euclidean distance
        let mut sum = F::zero();
        for (a_i, b_i) in a.iter().zip(b.iter()) {
            let diff = *a_i - *b_i;
            sum = sum + diff * diff;
        }
        sum.sqrt()
    });

    // Calculate centroids for each cluster
    let n_features = data.shape()[1];
    let mut centroids = Array2::<F>::zeros((n_clusters, n_features));
    let mut cluster_sizes = vec![0; n_clusters];

    for i in 0..n_samples {
        let label = labels[i];
        if label < 0 {
            continue; // Skip noise points
        }

        let cluster_idx = unique_labels.iter().position(|&l| l == label).unwrap();
        cluster_sizes[cluster_idx] += 1;

        for j in 0..n_features {
            centroids[[cluster_idx, j]] = centroids[[cluster_idx, j]] + data[[i, j]];
        }
    }

    // Normalize centroids by cluster sizes
    for i in 0..n_clusters {
        if cluster_sizes[i] == 0 {
            return Err(ClusteringError::ComputationError(format!(
                "Cluster {} is empty",
                unique_labels[i]
            )));
        }

        for j in 0..n_features {
            centroids[[i, j]] = centroids[[i, j]] / F::from_usize(cluster_sizes[i]).unwrap();
        }
    }

    // Calculate average distances within each cluster (cluster diameter)
    let mut cluster_diameters = vec![F::zero(); n_clusters];

    for i in 0..n_samples {
        let label = labels[i];
        if label < 0 {
            continue; // Skip noise points
        }

        let cluster_idx = unique_labels.iter().position(|&l| l == label).unwrap();
        let centroid = centroids.slice(s![cluster_idx, ..]);
        let point = data.slice(s![i, ..]);

        // Distance to centroid
        let dist = distance(&point.to_vec(), &centroid.to_vec());
        cluster_diameters[cluster_idx] = cluster_diameters[cluster_idx] + dist;
    }

    // Normalize by cluster sizes to get average distance
    for i in 0..n_clusters {
        cluster_diameters[i] = cluster_diameters[i] / F::from_usize(cluster_sizes[i]).unwrap();
    }

    // Calculate pairwise centroid distances
    let mut centroid_distances = Array2::<F>::zeros((n_clusters, n_clusters));
    for i in 0..n_clusters {
        for j in (i + 1)..n_clusters {
            let centroid_i = centroids.slice(s![i, ..]);
            let centroid_j = centroids.slice(s![j, ..]);
            let dist = distance(&centroid_i.to_vec(), &centroid_j.to_vec());
            centroid_distances[[i, j]] = dist;
            centroid_distances[[j, i]] = dist; // Symmetric
        }
    }

    // Compute Davies-Bouldin index
    let mut db_index = F::zero();
    for i in 0..n_clusters {
        let mut max_ratio = F::zero();
        for j in 0..n_clusters {
            if i != j {
                // Skip self-comparison
                let ratio =
                    (cluster_diameters[i] + cluster_diameters[j]) / centroid_distances[[i, j]];
                if ratio > max_ratio {
                    max_ratio = ratio;
                }
            }
        }
        db_index = db_index + max_ratio;
    }

    // Average over all clusters
    db_index = db_index / F::from_usize(n_clusters).unwrap();

    Ok(db_index)
}

/// Calculate the Calinski-Harabasz index for a clustering
///
/// The Calinski-Harabasz index (also known as the Variance Ratio Criterion) is defined as
/// the ratio of the between-cluster dispersion and the within-cluster dispersion.
/// Higher values indicate better clustering.
///
/// # Arguments
///
/// * `data` - The input data as a 2D array (n_samples x n_features)
/// * `labels` - The cluster labels for each sample (0 to k-1 for k clusters, -1 for noise)
///
/// # Returns
///
/// * `Result<F>` - The Calinski-Harabasz index value, or error if invalid input
///
/// # Examples
///
/// ```
/// use ndarray::{Array1, Array2};
/// use scirs2_cluster::metrics::calinski_harabasz_score;
///
/// // Example data with two clusters
/// let data = Array2::from_shape_vec((6, 2), vec![
///     1.0, 2.0,
///     1.2, 1.8,
///     0.8, 1.9,
///     5.0, 6.0,
///     5.2, 5.8,
///     4.8, 6.2,
/// ]).unwrap();
///
/// // Labels for the two clusters (0 and 1)
/// let labels = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);
///
/// // Calculate Calinski-Harabasz score
/// let score = calinski_harabasz_score(data.view(), labels.view()).unwrap();
/// println!("Calinski-Harabasz score: {}", score);
///
/// // Should be a high value for well-separated clusters
/// assert!(score > 10.0);
/// ```
pub fn calinski_harabasz_score<F: Float + FromPrimitive + Debug + PartialOrd>(
    data: ArrayView2<F>,
    labels: ArrayView1<i32>,
) -> Result<F> {
    // Check input dimensions
    if data.shape()[0] != labels.shape()[0] {
        return Err(ClusteringError::InvalidInput(
            "Number of samples in data and labels must match".into(),
        ));
    }

    let n_samples = data.shape()[0];
    if n_samples < 2 {
        return Err(ClusteringError::InvalidInput(
            "Calinski-Harabasz score requires at least 2 samples".into(),
        ));
    }

    // Find the unique labels (excluding noise points with label -1)
    let mut unique_labels = Vec::new();
    for &label in labels.iter() {
        if label >= 0 && !unique_labels.contains(&label) {
            unique_labels.push(label);
        }
    }

    let n_clusters = unique_labels.len();
    if n_clusters < 2 {
        return Err(ClusteringError::InvalidInput(
            "Calinski-Harabasz score requires at least 2 clusters".into(),
        ));
    }

    // Calculate the global centroid of the data
    let n_features = data.shape()[1];
    let mut global_centroid = Array1::<F>::zeros(n_features);
    let mut valid_samples = 0;

    for i in 0..n_samples {
        let label = labels[i];
        if label < 0 {
            continue; // Skip noise points
        }

        valid_samples += 1;
        for j in 0..n_features {
            global_centroid[j] = global_centroid[j] + data[[i, j]];
        }
    }

    if valid_samples == 0 {
        return Err(ClusteringError::ComputationError(
            "No valid samples (all are noise)".into(),
        ));
    }

    // Normalize global centroid
    for j in 0..n_features {
        global_centroid[j] = global_centroid[j] / F::from_usize(valid_samples).unwrap();
    }

    // Calculate centroids for each cluster
    let mut centroids = Array2::<F>::zeros((n_clusters, n_features));
    let mut cluster_sizes = vec![0; n_clusters];

    for i in 0..n_samples {
        let label = labels[i];
        if label < 0 {
            continue; // Skip noise points
        }

        let cluster_idx = unique_labels.iter().position(|&l| l == label).unwrap();
        cluster_sizes[cluster_idx] += 1;

        for j in 0..n_features {
            centroids[[cluster_idx, j]] = centroids[[cluster_idx, j]] + data[[i, j]];
        }
    }

    // Normalize centroids by cluster sizes
    for i in 0..n_clusters {
        if cluster_sizes[i] == 0 {
            return Err(ClusteringError::ComputationError(format!(
                "Cluster {} is empty",
                unique_labels[i]
            )));
        }

        for j in 0..n_features {
            centroids[[i, j]] = centroids[[i, j]] / F::from_usize(cluster_sizes[i]).unwrap();
        }
    }

    // Calculate between-clusters sum of squares (BCSS)
    let mut bcss = F::zero();
    for i in 0..n_clusters {
        let cluster_size = F::from_usize(cluster_sizes[i]).unwrap();
        let mut squared_dist = F::zero();

        for j in 0..n_features {
            let diff = centroids[[i, j]] - global_centroid[j];
            squared_dist = squared_dist + diff * diff;
        }

        bcss = bcss + cluster_size * squared_dist;
    }

    // Calculate within-clusters sum of squares (WCSS)
    let mut wcss = F::zero();
    for i in 0..n_samples {
        let label = labels[i];
        if label < 0 {
            continue; // Skip noise points
        }

        let cluster_idx = unique_labels.iter().position(|&l| l == label).unwrap();
        let mut squared_dist = F::zero();

        for j in 0..n_features {
            let diff = data[[i, j]] - centroids[[cluster_idx, j]];
            squared_dist = squared_dist + diff * diff;
        }

        wcss = wcss + squared_dist;
    }

    // Calculate the Calinski-Harabasz index
    let ch_index = if wcss > F::zero() {
        // (BCSS / (k-1)) / (WCSS / (n-k))
        (bcss / F::from_usize(n_clusters - 1).unwrap())
            / (wcss / F::from_usize(valid_samples - n_clusters).unwrap())
    } else {
        // In the unlikely case where all points are at the exact centroid of their clusters
        F::infinity()
    };

    Ok(ch_index)
}

/// Calculate the silhouette scores for each sample in the dataset
///
/// # Arguments
///
/// * `data` - The input data as a 2D array (n_samples x n_features)
/// * `labels` - The cluster labels for each sample (0 to k-1 for k clusters, -1 for noise)
/// * `metric` - Optional distance metric function: fn(&[F], &[F]) -> F
///
/// # Returns
///
/// An array of silhouette values for each sample (non-noise points), or error if invalid input
pub fn silhouette_samples<F: Float + FromPrimitive + Debug + PartialOrd>(
    data: ArrayView2<F>,
    labels: ArrayView1<i32>,
    metric: Option<fn(&[F], &[F]) -> F>,
) -> Result<Array1<F>> {
    // Check input dimensions
    if data.shape()[0] != labels.shape()[0] {
        return Err(ClusteringError::InvalidInput(
            "Number of samples in data and labels must match".into(),
        ));
    }

    let n_samples = data.shape()[0];
    if n_samples < 2 {
        return Err(ClusteringError::InvalidInput(
            "Silhouette score requires at least 2 samples".into(),
        ));
    }

    // Find the unique labels (excluding noise points with label -1)
    let mut unique_labels = Vec::new();
    for &label in labels.iter() {
        if label >= 0 && !unique_labels.contains(&label) {
            unique_labels.push(label);
        }
    }

    let n_clusters = unique_labels.len();
    if n_clusters < 2 || n_clusters >= n_samples {
        return Err(ClusteringError::InvalidInput(
            "Silhouette score requires at least 2 clusters and fewer clusters than samples".into(),
        ));
    }

    // Choose distance metric function
    let distance = metric.unwrap_or(|a, b| {
        // Default to Euclidean distance
        let mut sum = F::zero();
        for (a_i, b_i) in a.iter().zip(b.iter()) {
            let diff = *a_i - *b_i;
            sum = sum + diff * diff;
        }
        sum.sqrt()
    });

    // Calculate pairwise distances
    let mut distances = Array2::<F>::zeros((n_samples, n_samples));
    for i in 0..n_samples {
        for j in (i + 1)..n_samples {
            let dist = distance(&data.row(i).to_vec(), &data.row(j).to_vec());
            distances[[i, j]] = dist;
            distances[[j, i]] = dist; // Distance matrix is symmetric
        }
    }

    // For each sample, calculate silhouette value
    let mut silhouette_values = vec![F::zero(); n_samples];

    for i in 0..n_samples {
        let sample_label = labels[i];

        // Skip noise points (label -1)
        if sample_label < 0 {
            continue;
        }

        // Find all samples in the same cluster
        let mut same_cluster = Vec::new();
        for j in 0..n_samples {
            if i != j && labels[j] == sample_label {
                same_cluster.push(j);
            }
        }

        // If alone in cluster, silhouette is 0 (can't calculate intra-cluster distance)
        if same_cluster.is_empty() {
            silhouette_values[i] = F::zero();
            continue;
        }

        // Calculate mean distance to all points in the same cluster (a)
        let mut intra_cluster_dist_sum = F::zero();
        for &j in &same_cluster {
            intra_cluster_dist_sum = intra_cluster_dist_sum + distances[[i, j]];
        }
        let a = intra_cluster_dist_sum / F::from_usize(same_cluster.len()).unwrap();

        // Find the mean distance to the closest neighboring cluster (b)
        let mut min_inter_cluster_dist = F::infinity();

        for &neighbor_label in &unique_labels {
            if neighbor_label == sample_label {
                continue;
            }

            // Calculate mean distance to this neighboring cluster
            let mut inter_cluster_dist_sum = F::zero();
            let mut count = 0;

            for j in 0..n_samples {
                if labels[j] == neighbor_label {
                    inter_cluster_dist_sum = inter_cluster_dist_sum + distances[[i, j]];
                    count += 1;
                }
            }

            if count > 0 {
                let mean_dist = inter_cluster_dist_sum / F::from_usize(count).unwrap();
                if mean_dist < min_inter_cluster_dist {
                    min_inter_cluster_dist = mean_dist;
                }
            }
        }

        let b = min_inter_cluster_dist;

        // Calculate silhouette value: (b - a) / max(a, b)
        silhouette_values[i] = (b - a) / a.max(b);
    }

    Ok(Array1::from(silhouette_values))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_silhouette_score_well_separated() {
        // Two well-separated clusters
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 2.0, 1.2, 1.8, 0.8, 1.9, 10.0, 10.0, 10.2, 9.8, 9.9, 10.1,
            ],
        )
        .unwrap();

        let labels = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);

        let score = silhouette_score(data.view(), labels.view(), None).unwrap();
        assert!(
            score > 0.9,
            "Silhouette score should be high for well-separated clusters"
        );
    }

    #[test]
    fn test_silhouette_score_overlapping() {
        // Two overlapping clusters
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 2.0, 1.2, 1.8, 3.0, 3.0, // Overlapping point
                2.5, 2.5, // Overlapping point
                4.0, 4.0, 4.2, 3.8,
            ],
        )
        .unwrap();

        let labels = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);

        let score = silhouette_score(data.view(), labels.view(), None).unwrap();
        assert!(
            score < 0.9,
            "Silhouette score should be lower for overlapping clusters"
        );
    }

    #[test]
    fn test_silhouette_samples() {
        // Simple dataset
        let data =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 1.0, 4.0, 4.0, 5.0, 4.0]).unwrap();

        let labels = Array1::from_vec(vec![0, 0, 1, 1]);

        let samples = silhouette_samples(data.view(), labels.view(), None).unwrap();

        // Check dimensions
        assert_eq!(samples.len(), 4);

        // All samples should have positive silhouette scores
        for &score in samples.iter() {
            assert!(score > 0.0);
        }
    }

    #[test]
    fn test_davies_bouldin_score_well_separated() {
        // Two well-separated clusters
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 2.0, 1.2, 1.8, 0.8, 1.9, 10.0, 10.0, 10.2, 9.8, 9.9, 10.1,
            ],
        )
        .unwrap();

        let labels = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);

        let score = davies_bouldin_score(data.view(), labels.view(), None).unwrap();
        // Low score is better for Davies-Bouldin index
        assert!(
            score < 0.5,
            "Davies-Bouldin score should be low for well-separated clusters"
        );
    }

    #[test]
    fn test_davies_bouldin_score_overlapping() {
        // Two overlapping clusters
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 2.0, 1.2, 1.8, 3.0, 3.0, // Overlapping point
                2.5, 2.5, // Overlapping point
                4.0, 4.0, 4.2, 3.8,
            ],
        )
        .unwrap();

        let labels = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);

        let score = davies_bouldin_score(data.view(), labels.view(), None).unwrap();
        // Higher score for overlapping clusters
        assert!(
            score > 0.5,
            "Davies-Bouldin score should be higher for overlapping clusters"
        );
    }

    #[test]
    fn test_calinski_harabasz_score_well_separated() {
        // Two well-separated clusters
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 2.0, 1.2, 1.8, 0.8, 1.9, 10.0, 10.0, 10.2, 9.8, 9.9, 10.1,
            ],
        )
        .unwrap();

        let labels = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);

        let score = calinski_harabasz_score(data.view(), labels.view()).unwrap();
        // High score is better for Calinski-Harabasz index
        assert!(
            score > 100.0,
            "Calinski-Harabasz score should be high for well-separated clusters"
        );
    }

    #[test]
    fn test_calinski_harabasz_score_overlapping() {
        // Two overlapping clusters
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 2.0, 1.2, 1.8, 3.0, 3.0, // Overlapping point
                2.5, 2.5, // Overlapping point
                4.0, 4.0, 4.2, 3.8,
            ],
        )
        .unwrap();

        let labels = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);

        let score = calinski_harabasz_score(data.view(), labels.view()).unwrap();
        // Lower score for overlapping clusters compared to well-separated ones
        assert!(
            score < 100.0,
            "Calinski-Harabasz score should be lower for overlapping clusters"
        );
    }
}
