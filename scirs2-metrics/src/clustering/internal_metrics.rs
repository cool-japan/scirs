//! Internal clustering metrics
//!
//! This module provides functions for evaluating clustering algorithms using internal metrics,
//! which assess clustering quality without external ground truth. These include
//! silhouette score, Davies-Bouldin index, Calinski-Harabasz index, and Dunn index.

use ndarray::{Array1, Array2, ArrayBase, Data, Dimension, Ix2};
use num_traits::{Float, NumCast};
use std::collections::HashMap;

use super::{calculate_distance, group_by_labels, pairwise_distances};
use crate::error::{MetricsError, Result};

/// Structure containing detailed silhouette analysis results
#[derive(Debug, Clone)]
pub struct SilhouetteAnalysis<F: Float> {
    /// Sample-wise silhouette scores
    pub sample_values: Vec<F>,

    /// Mean silhouette score for all samples
    pub mean_score: F,

    /// Mean silhouette score for each cluster
    pub cluster_scores: HashMap<usize, F>,

    /// Sorted indices for visualization (samples ordered by cluster and silhouette value)
    pub sorted_indices: Vec<usize>,

    /// Original cluster labels mapped to consecutive integers (for visualization)
    pub cluster_mapping: HashMap<usize, usize>,

    /// Samples per cluster (ordered by cluster_mapping)
    pub cluster_sizes: Vec<usize>,
}

/// Calculates the silhouette score for a clustering
///
/// The silhouette score measures how similar an object is to its own cluster
/// compared to other clusters. The silhouette score ranges from -1 to 1, where
/// a high value indicates that the object is well matched to its own cluster
/// and poorly matched to neighboring clusters.
///
/// # Arguments
///
/// * `x` - Array of shape (n_samples, n_features) - The data
/// * `labels` - Array of shape (n_samples,) - Predicted labels for each sample
/// * `metric` - Distance metric to use. Currently only 'euclidean' is supported.
///
/// # Returns
///
/// * The mean silhouette coefficient for all samples
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array2};
/// use scirs2_metrics::clustering::silhouette_score;
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
/// let score = silhouette_score(&x, &labels, "euclidean").unwrap();
/// assert!(score > 0.8); // High score for well-separated clusters
/// ```
pub fn silhouette_score<F, S1, S2, D>(
    x: &ArrayBase<S1, Ix2>,
    labels: &ArrayBase<S2, D>,
    metric: &str,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug + ndarray::ScalarOperand + 'static,
    S1: Data<Elem = F>,
    S2: Data<Elem = usize>,
    D: Dimension,
{
    let silhouette_results = silhouette_analysis(x, labels, metric)?;
    Ok(silhouette_results.mean_score)
}

/// Calculate silhouette samples for a clustering
///
/// Returns the silhouette score for each sample in the clustering, which can be
/// useful for more detailed analysis than just the mean silhouette score.
///
/// # Arguments
///
/// * `x` - Array of shape (n_samples, n_features) - The data
/// * `labels` - Array of shape (n_samples,) - Predicted labels for each sample
/// * `metric` - Distance metric to use. Currently only 'euclidean' is supported.
///
/// # Returns
///
/// * Array of shape (n_samples,) containing the silhouette score for each sample
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array2};
/// use scirs2_metrics::clustering::silhouette_samples;
///
/// // Create a small dataset with 2 clusters
/// let x = Array2::from_shape_vec((6, 2), vec![
///     1.0, 2.0, 1.5, 1.8, 1.2, 2.2, // Cluster 0
///     5.0, 6.0, 5.2, 5.8, 5.5, 6.2, // Cluster 1
/// ]).unwrap();
///
/// let labels = array![0, 0, 0, 1, 1, 1];
///
/// let samples = silhouette_samples(&x, &labels, "euclidean").unwrap();
/// assert_eq!(samples.len(), 6);
///
/// // Calculate the mean silhouette score manually
/// let mean_score = samples.iter().sum::<f64>() / samples.len() as f64;
/// assert!(mean_score > 0.8); // High score for well-separated clusters
/// ```
pub fn silhouette_samples<F, S1, S2, D>(
    x: &ArrayBase<S1, Ix2>,
    labels: &ArrayBase<S2, D>,
    metric: &str,
) -> Result<Vec<F>>
where
    F: Float + NumCast + std::fmt::Debug + ndarray::ScalarOperand + 'static,
    S1: Data<Elem = F>,
    S2: Data<Elem = usize>,
    D: Dimension,
{
    let analysis = silhouette_analysis(x, labels, metric)?;
    Ok(analysis.sample_values)
}

/// Calculate silhouette scores per cluster
///
/// Returns the mean silhouette score for each cluster, allowing you to
/// identify which clusters are more cohesive than others.
///
/// # Arguments
///
/// * `x` - Array of shape (n_samples, n_features) - The data
/// * `labels` - Array of shape (n_samples,) - Predicted labels for each sample
/// * `metric` - Distance metric to use. Currently only 'euclidean' is supported.
///
/// # Returns
///
/// * HashMap mapping cluster labels to their mean silhouette scores
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array2};
/// use scirs2_metrics::clustering::silhouette_scores_per_cluster;
///
/// // Create a small dataset with 3 clusters
/// let x = Array2::from_shape_vec((9, 2), vec![
///     1.0, 2.0, 1.5, 1.8, 1.2, 2.2,  // Cluster 0
///     5.0, 6.0, 5.2, 5.8, 5.5, 6.2,  // Cluster 1
///     9.0, 10.0, 9.2, 9.8, 9.5, 10.2, // Cluster 2
/// ]).unwrap();
///
/// let labels = array![0, 0, 0, 1, 1, 1, 2, 2, 2];
///
/// let cluster_scores = silhouette_scores_per_cluster(&x, &labels, "euclidean").unwrap();
/// assert_eq!(cluster_scores.len(), 3);
/// assert!(cluster_scores[&0] > 0.5);
/// assert!(cluster_scores[&1] > 0.5);
/// assert!(cluster_scores[&2] > 0.5);
/// ```
pub fn silhouette_scores_per_cluster<F, S1, S2, D>(
    x: &ArrayBase<S1, Ix2>,
    labels: &ArrayBase<S2, D>,
    metric: &str,
) -> Result<HashMap<usize, F>>
where
    F: Float + NumCast + std::fmt::Debug + ndarray::ScalarOperand + 'static,
    S1: Data<Elem = F>,
    S2: Data<Elem = usize>,
    D: Dimension,
{
    let analysis = silhouette_analysis(x, labels, metric)?;
    Ok(analysis.cluster_scores)
}

/// Calculates detailed silhouette information for a clustering
///
/// This function provides sample-wise silhouette scores, cluster-wise averages,
/// and ordering information for visualization. It's an enhanced version of
/// silhouette_score that returns more detailed information.
///
/// # Arguments
///
/// * `x` - Array of shape (n_samples, n_features) - The data
/// * `labels` - Array of shape (n_samples,) - Predicted labels for each sample
/// * `metric` - Distance metric to use. Currently only 'euclidean' is supported.
///
/// # Returns
///
/// * `SilhouetteAnalysis` struct containing detailed silhouette information
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array2};
/// use scirs2_metrics::clustering::silhouette_analysis;
///
/// // Create a small dataset with 3 clusters
/// let x = Array2::from_shape_vec((9, 2), vec![
///     1.0, 2.0, 1.5, 1.8, 1.2, 2.2,  // Cluster 0
///     5.0, 6.0, 5.2, 5.8, 5.5, 6.2,  // Cluster 1
///     9.0, 10.0, 9.2, 9.8, 9.5, 10.2, // Cluster 2
/// ]).unwrap();
///
/// let labels = array![0, 0, 0, 1, 1, 1, 2, 2, 2];
///
/// let analysis = silhouette_analysis(&x, &labels, "euclidean").unwrap();
///
/// // Get overall silhouette score
/// let score = analysis.mean_score;
/// assert!(score > 0.8); // High score for well-separated clusters
///
/// // Get cluster-wise silhouette scores
/// for (cluster, score) in &analysis.cluster_scores {
///     println!("Cluster {} silhouette score: {}", cluster, score);
/// }
///
/// // Access individual sample silhouette values
/// for (i, &value) in analysis.sample_values.iter().enumerate() {
///     println!("Sample {} silhouette value: {}", i, value);
/// }
/// ```
pub fn silhouette_analysis<F, S1, S2, D>(
    x: &ArrayBase<S1, Ix2>,
    labels: &ArrayBase<S2, D>,
    metric: &str,
) -> Result<SilhouetteAnalysis<F>>
where
    F: Float + NumCast + std::fmt::Debug + ndarray::ScalarOperand + 'static,
    S1: Data<Elem = F>,
    S2: Data<Elem = usize>,
    D: Dimension,
{
    // Check that the metric is supported
    if metric != "euclidean" {
        return Err(MetricsError::InvalidInput(format!(
            "Unsupported metric: {}. Only 'euclidean' is currently supported.",
            metric
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

    // Check that there are at least 2 samples
    if n_samples < 2 {
        return Err(MetricsError::InvalidInput(
            "n_samples must be at least 2".to_string(),
        ));
    }

    // Group samples by label
    let clusters = group_by_labels(x, labels)?;

    // Check that there are at least 2 clusters
    if clusters.len() < 2 {
        return Err(MetricsError::InvalidInput(
            "Number of labels is 1. Silhouette analysis is undefined for a single cluster."
                .to_string(),
        ));
    }

    // Check that all clusters have at least 1 sample
    let empty_clusters: Vec<_> = clusters
        .iter()
        .filter(|(_, samples)| samples.is_empty())
        .map(|(&label, _)| label)
        .collect();

    if !empty_clusters.is_empty() {
        return Err(MetricsError::InvalidInput(format!(
            "Empty clusters found: {:?}",
            empty_clusters
        )));
    }

    // Compute distance matrix (more efficient than recomputing distances)
    let distances = pairwise_distances(x, metric)?;

    // Compute silhouette scores for each sample
    let mut silhouette_values = Vec::with_capacity(n_samples);
    let mut sample_clusters = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let label_i = labels.iter().nth(i).ok_or_else(|| {
            MetricsError::InvalidInput(format!("Could not access index {} in labels", i))
        })?;
        let cluster_i = &clusters[label_i];
        sample_clusters.push(*label_i);

        // Calculate the mean intra-cluster distance (a)
        let mut a = F::zero();
        let mut count_a = 0;

        for &j in cluster_i {
            if i == j {
                continue;
            }
            a = a + distances[[i, j]];
            count_a += 1;
        }

        // Handle single sample in cluster (set a to 0)
        if count_a > 0 {
            a = a / F::from(count_a).unwrap();
        }

        // Calculate the mean nearest-cluster distance (b)
        let mut b = None;

        for (label_j, cluster_j) in &clusters {
            if *label_j == *label_i {
                continue;
            }

            // Calculate mean distance to this cluster
            let mut cluster_dist = F::zero();
            for &j in cluster_j {
                cluster_dist = cluster_dist + distances[[i, j]];
            }
            let cluster_dist = cluster_dist / F::from(cluster_j.len()).unwrap();

            // Update b if this is the closest cluster
            if let Some(current_b) = b {
                if cluster_dist < current_b {
                    b = Some(cluster_dist);
                }
            } else {
                b = Some(cluster_dist);
            }
        }

        // Calculate silhouette score
        let s = if let Some(b) = b {
            if a < b {
                F::one() - a / b
            } else if a > b {
                b / a - F::one()
            } else {
                F::zero()
            }
        } else {
            F::zero() // Will never happen if there are at least 2 clusters
        };

        silhouette_values.push(s);
    }

    // Calculate mean silhouette score
    let sum = silhouette_values
        .iter()
        .fold(F::zero(), |acc, &val| acc + val);
    let mean_score = sum / F::from(n_samples).unwrap();

    // Calculate cluster-wise silhouette scores
    let mut cluster_scores = HashMap::new();
    for (label, indices) in &clusters {
        let mut cluster_sum = F::zero();
        for &idx in indices {
            cluster_sum = cluster_sum + silhouette_values[idx];
        }
        let cluster_mean = cluster_sum / F::from(indices.len()).unwrap();
        cluster_scores.insert(*label, cluster_mean);
    }

    // Create a mapping from original cluster labels to consecutive integers (for visualization)
    let unique_labels: Vec<_> = clusters.keys().cloned().collect();
    let mut cluster_mapping = HashMap::new();
    for (i, &label) in unique_labels.iter().enumerate() {
        cluster_mapping.insert(label, i);
    }

    // Create list of cluster sizes
    let mut cluster_sizes = vec![0; cluster_mapping.len()];
    for (label, indices) in &clusters {
        let mapped_idx = cluster_mapping[label];
        cluster_sizes[mapped_idx] = indices.len();
    }

    // Create sorted indices for visualization
    // First, sort by cluster mapping
    // Then, within each cluster, sort by silhouette value (descending)
    let mut samples_with_scores: Vec<(usize, F, usize)> = silhouette_values
        .iter()
        .enumerate()
        .map(|(i, &s)| (i, s, cluster_mapping[&sample_clusters[i]]))
        .collect();

    // Sort by cluster (ascending), then by silhouette value (descending)
    samples_with_scores.sort_by(|a, b| a.2.cmp(&b.2).then(b.1.partial_cmp(&a.1).unwrap()));

    let sorted_indices = samples_with_scores.iter().map(|&(i, _, _)| i).collect();

    Ok(SilhouetteAnalysis {
        sample_values: silhouette_values,
        mean_score,
        cluster_scores,
        sorted_indices,
        cluster_mapping,
        cluster_sizes,
    })
}

/// Calculates the Davies-Bouldin index for a clustering
///
/// The Davies-Bouldin index measures the average similarity between clusters,
/// where the similarity is a ratio of within-cluster distances to between-cluster distances.
/// The lower the value, the better the clustering.
///
/// # Arguments
///
/// * `x` - Array of shape (n_samples, n_features) - The data
/// * `labels` - Array of shape (n_samples,) - Predicted labels for each sample
///
/// # Returns
///
/// * The Davies-Bouldin index
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array2};
/// use scirs2_metrics::clustering::davies_bouldin_score;
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
/// let score = davies_bouldin_score(&x, &labels).unwrap();
/// assert!(score < 0.5); // Low score for well-separated clusters
/// ```
pub fn davies_bouldin_score<F, S1, S2, D>(
    x: &ArrayBase<S1, Ix2>,
    labels: &ArrayBase<S2, D>,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug + ndarray::ScalarOperand + 'static,
    S1: Data<Elem = F>,
    S2: Data<Elem = usize>,
    D: Dimension,
{
    // Check that x and labels have the same number of samples
    let n_samples = x.shape()[0];
    if n_samples != labels.len() {
        return Err(MetricsError::InvalidInput(format!(
            "x has {} samples, but labels has {} samples",
            n_samples,
            labels.len()
        )));
    }

    // Check that there are at least 2 samples
    if n_samples < 2 {
        return Err(MetricsError::InvalidInput(
            "n_samples must be at least 2".to_string(),
        ));
    }

    // Group samples by label
    let clusters = group_by_labels(x, labels)?;

    // Check that there are at least 2 clusters
    if clusters.len() < 2 {
        return Err(MetricsError::InvalidInput(
            "Number of labels is 1. Davies-Bouldin index is undefined for a single cluster."
                .to_string(),
        ));
    }

    // Compute centroids for each cluster
    let mut centroids = HashMap::new();
    for (&label, indices) in &clusters {
        let mut centroid = Array1::<F>::zeros(x.shape()[1]);
        for &idx in indices {
            centroid = centroid + x.row(idx).to_owned();
        }
        centroid = centroid / F::from(indices.len()).unwrap();
        centroids.insert(label, centroid);
    }

    // Compute average distance to centroid for each cluster
    let mut avg_distances = HashMap::new();
    for (&label, indices) in &clusters {
        let centroid = centroids.get(&label).unwrap();
        let mut total_distance = F::zero();
        for &idx in indices {
            total_distance = total_distance
                + calculate_distance(&x.row(idx).to_vec(), &centroid.to_vec(), "euclidean")?;
        }
        let avg_distance = total_distance / F::from(indices.len()).unwrap();
        avg_distances.insert(label, avg_distance);
    }

    // Compute Davies-Bouldin index
    let mut db_index = F::zero();
    let labels_vec: Vec<_> = clusters.keys().cloned().collect();

    for i in 0..labels_vec.len() {
        let label_i = labels_vec[i];
        let centroid_i = centroids.get(&label_i).unwrap();
        let avg_dist_i = avg_distances.get(&label_i).unwrap();

        let mut max_ratio = F::zero();
        for (j, &label_j) in labels_vec.iter().enumerate() {
            if i == j {
                continue;
            }
            let centroid_j = centroids.get(&label_j).unwrap();
            let avg_dist_j = avg_distances.get(&label_j).unwrap();

            // Distance between centroids
            let centroid_dist =
                calculate_distance(&centroid_i.to_vec(), &centroid_j.to_vec(), "euclidean")?;

            // Ratio of sum of intra-cluster distances to inter-cluster distance
            let ratio = (*avg_dist_i + *avg_dist_j) / centroid_dist;

            // Update max ratio
            if ratio > max_ratio {
                max_ratio = ratio;
            }
        }

        db_index = db_index + max_ratio;
    }

    // Normalize by number of clusters
    Ok(db_index / F::from(labels_vec.len()).unwrap())
}

/// Calculates the Calinski-Harabasz index (Variance Ratio Criterion)
///
/// The Calinski-Harabasz index is defined as the ratio of the between-clusters
/// dispersion and the within-cluster dispersion. Higher values indicate better clustering.
///
/// # Arguments
///
/// * `x` - Array of shape (n_samples, n_features) - The data
/// * `labels` - Array of shape (n_samples,) - Predicted labels for each sample
///
/// # Returns
///
/// * The Calinski-Harabasz index
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array2};
/// use scirs2_metrics::clustering::calinski_harabasz_score;
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
/// let score = calinski_harabasz_score(&x, &labels).unwrap();
/// assert!(score > 50.0); // High score for well-separated clusters
/// ```
pub fn calinski_harabasz_score<F, S1, S2, D>(
    x: &ArrayBase<S1, Ix2>,
    labels: &ArrayBase<S2, D>,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug + ndarray::ScalarOperand + 'static,
    S1: Data<Elem = F>,
    S2: Data<Elem = usize>,
    D: Dimension,
{
    // Check that x and labels have the same number of samples
    let n_samples = x.shape()[0];
    if n_samples != labels.len() {
        return Err(MetricsError::InvalidInput(format!(
            "x has {} samples, but labels has {} samples",
            n_samples,
            labels.len()
        )));
    }

    // Check that there are at least 2 samples
    if n_samples < 2 {
        return Err(MetricsError::InvalidInput(
            "n_samples must be at least 2".to_string(),
        ));
    }

    // Group samples by label
    let clusters = group_by_labels(x, labels)?;

    // Check that there are at least 2 clusters
    if clusters.len() < 2 {
        return Err(MetricsError::InvalidInput(
            "Number of labels is 1. Calinski-Harabasz index is undefined for a single cluster."
                .to_string(),
        ));
    }

    // Compute the global centroid
    let mut global_centroid = Array1::<F>::zeros(x.shape()[1]);
    for i in 0..n_samples {
        global_centroid = global_centroid + x.row(i).to_owned();
    }
    global_centroid = global_centroid / F::from(n_samples).unwrap();

    // Compute centroids for each cluster
    let mut centroids = HashMap::new();
    for (&label, indices) in &clusters {
        let mut centroid = Array1::<F>::zeros(x.shape()[1]);
        for &idx in indices {
            centroid = centroid + x.row(idx).to_owned();
        }
        centroid = centroid / F::from(indices.len()).unwrap();
        centroids.insert(label, centroid);
    }

    // Compute between-cluster dispersion
    let mut between_disp = F::zero();
    for (label, indices) in &clusters {
        let cluster_size = F::from(indices.len()).unwrap();
        let centroid = centroids.get(label).unwrap();

        // Calculate squared distance between cluster centroid and global centroid
        let mut squared_dist = F::zero();
        for (c, g) in centroid.iter().zip(global_centroid.iter()) {
            let diff = *c - *g;
            squared_dist = squared_dist + diff * diff;
        }

        between_disp = between_disp + cluster_size * squared_dist;
    }

    // Compute within-cluster dispersion
    let mut within_disp = F::zero();
    for (label, indices) in &clusters {
        let centroid = centroids.get(label).unwrap();

        let mut cluster_disp = F::zero();
        for &idx in indices {
            let mut squared_dist = F::zero();
            for (x_val, c_val) in x.row(idx).iter().zip(centroid.iter()) {
                let diff = *x_val - *c_val;
                squared_dist = squared_dist + diff * diff;
            }
            cluster_disp = cluster_disp + squared_dist;
        }

        within_disp = within_disp + cluster_disp;
    }

    // Handle edge cases
    if within_disp <= F::epsilon() {
        return Err(MetricsError::CalculationError(
            "Within-cluster dispersion is zero".to_string(),
        ));
    }

    // Calculate Calinski-Harabasz index
    let n_clusters = F::from(clusters.len()).unwrap();
    Ok(
        between_disp * (F::from(n_samples - clusters.len()).unwrap())
            / (within_disp * (n_clusters - F::one())),
    )
}

/// Calculates the Dunn index for a clustering
///
/// The Dunn index is the ratio of the minimum inter-cluster distance to the maximum
/// intra-cluster distance. Higher values indicate better clustering.
///
/// # Arguments
///
/// * `x` - Array of shape (n_samples, n_features) - The data
/// * `labels` - Array of shape (n_samples,) - Predicted labels for each sample
///
/// # Returns
///
/// * The Dunn index
pub fn dunn_index<F, S1, S2, D>(x: &ArrayBase<S1, Ix2>, labels: &ArrayBase<S2, D>) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug + 'static,
    S1: Data<Elem = F>,
    S2: Data<Elem = usize>,
    D: Dimension,
{
    // Check that x and labels have the same number of samples
    let n_samples = x.shape()[0];
    if n_samples != labels.len() {
        return Err(MetricsError::InvalidInput(format!(
            "x has {} samples, but labels has {} samples",
            n_samples,
            labels.len()
        )));
    }

    // Check that there are at least 2 samples
    if n_samples < 2 {
        return Err(MetricsError::InvalidInput(
            "n_samples must be at least 2".to_string(),
        ));
    }

    // Group samples by label
    let clusters = group_by_labels(x, labels)?;

    // Check that there are at least 2 clusters
    if clusters.len() < 2 {
        return Err(MetricsError::InvalidInput(
            "Number of labels is 1. Dunn index is undefined for a single cluster.".to_string(),
        ));
    }

    // Compute distance matrix
    let distances = pairwise_distances(x, "euclidean")?;

    // Calculate maximum intra-cluster distance for each cluster
    let mut max_intra_cluster_distance = F::zero();
    for indices in clusters.values() {
        let mut max_distance = F::zero();
        for (i, &idx1) in indices.iter().enumerate() {
            for &idx2 in &indices[i + 1..] {
                let dist = distances[[idx1, idx2]];
                if dist > max_distance {
                    max_distance = dist;
                }
            }
        }
        if max_distance > max_intra_cluster_distance {
            max_intra_cluster_distance = max_distance;
        }
    }

    if max_intra_cluster_distance <= F::epsilon() {
        return Err(MetricsError::CalculationError(
            "Maximum intra-cluster distance is zero".to_string(),
        ));
    }

    // Calculate minimum inter-cluster distance
    let mut min_inter_cluster_distance = F::infinity();
    let cluster_labels: Vec<_> = clusters.keys().collect();

    for i in 0..cluster_labels.len() {
        for j in i + 1..cluster_labels.len() {
            let cluster_i = &clusters[cluster_labels[i]];
            let cluster_j = &clusters[cluster_labels[j]];

            let mut min_distance = F::infinity();
            for &idx1 in cluster_i {
                for &idx2 in cluster_j {
                    let dist = distances[[idx1, idx2]];
                    if dist < min_distance {
                        min_distance = dist;
                    }
                }
            }

            if min_distance < min_inter_cluster_distance {
                min_inter_cluster_distance = min_distance;
            }
        }
    }

    // Calculate Dunn index
    Ok(min_inter_cluster_distance / max_intra_cluster_distance)
}
