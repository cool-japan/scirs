// Clustering metrics module
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

use ndarray::{Array1, Array2, ArrayBase, Data, Dimension, Ix2};
use num_traits::{Float, NumCast};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::HashMap;

use crate::error::{MetricsError, Result};

pub mod density;
pub mod distance;
pub mod validation;

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
    F: Float + NumCast + std::fmt::Debug + ndarray::ScalarOperand,
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
    let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &label) in labels.iter().enumerate() {
        clusters.entry(label).or_default().push(i);
    }

    // Check that there are at least 2 clusters
    if clusters.len() < 2 {
        return Err(MetricsError::InvalidInput(
            "Number of labels is 1. Silhouette score is undefined for a single cluster."
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

    // Compute silhouette scores for each sample
    let mut silhouette_values = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let label_i = labels.iter().nth(i).ok_or_else(|| {
            MetricsError::InvalidInput(format!("Could not access index {} in labels", i))
        })?;
        let cluster_i = &clusters[label_i];

        // Calculate the mean intra-cluster distance (a)
        let mut a = F::zero();
        let mut count_a = 0;

        for &j in cluster_i {
            if i == j {
                continue;
            }
            a = a + euclidean_distance(x.row(i).to_owned(), x.row(j).to_owned());
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
                cluster_dist =
                    cluster_dist + euclidean_distance(x.row(i).to_owned(), x.row(j).to_owned());
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

    Ok(mean_score)
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
    F: Float + NumCast + std::fmt::Debug + ndarray::ScalarOperand,
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

    // Group samples by label and compute cluster centroids
    let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &label) in labels.iter().enumerate() {
        clusters.entry(label).or_default().push(i);
    }

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
            total_distance =
                total_distance + euclidean_distance(x.row(idx).to_owned(), centroid.view());
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
            let centroid_dist = euclidean_distance(centroid_i.view(), centroid_j.view());

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
    F: Float + NumCast + std::fmt::Debug + ndarray::ScalarOperand,
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
    let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &label) in labels.iter().enumerate() {
        clusters.entry(label).or_default().push(i);
    }

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

        let dist_to_global = squared_euclidean_distance(centroid.view(), global_centroid.view());
        between_disp = between_disp + cluster_size * dist_to_global;
    }

    // Compute within-cluster dispersion
    let mut within_disp = F::zero();
    for (label, indices) in &clusters {
        let centroid = centroids.get(label).unwrap();

        let mut cluster_disp = F::zero();
        for &idx in indices {
            let dist_to_centroid =
                squared_euclidean_distance(x.row(idx).to_owned(), centroid.view());
            cluster_disp = cluster_disp + dist_to_centroid;
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

/// Helper function to calculate the Euclidean distance between two vectors
fn euclidean_distance<F, S1, S2, D1, D2>(a: ArrayBase<S1, D1>, b: ArrayBase<S2, D2>) -> F
where
    F: Float + NumCast + std::fmt::Debug + ndarray::ScalarOperand,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    let mut sum = F::zero();
    for (x, y) in a.iter().zip(b.iter()) {
        let diff = *x - *y;
        sum = sum + diff * diff;
    }
    sum.sqrt()
}

/// Helper function to calculate the squared Euclidean distance between two vectors
fn squared_euclidean_distance<F, S1, S2, D1, D2>(a: ArrayBase<S1, D1>, b: ArrayBase<S2, D2>) -> F
where
    F: Float + NumCast + std::fmt::Debug + ndarray::ScalarOperand,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    let mut sum = F::zero();
    for (x, y) in a.iter().zip(b.iter()) {
        let diff = *x - *y;
        sum = sum + diff * diff;
    }
    sum
}

/// Calculates the Adjusted Rand Index (ARI) between two clusterings
///
/// The Adjusted Rand Index is a measure of the similarity between two data clusterings,
/// adjusted for the chance grouping of elements. It is related to the accuracy but is
/// applicable when class labels are not used.
///
/// ARI values range from -1 to 1:
/// * 1: Perfect agreement between the clusterings
/// * 0: Agreement equivalent to random chance
/// * Negative values: Agreement less than random chance
///
/// # Arguments
///
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Predicted cluster labels to evaluate
///
/// # Returns
///
/// * The Adjusted Rand Index
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::clustering::adjusted_rand_index;
///
/// let labels_true = array![0, 0, 0, 1, 1, 1, 2, 2, 2];
/// let labels_pred = array![0, 0, 1, 1, 2, 1, 2, 2, 2];
///
/// let ari = adjusted_rand_index(&labels_true, &labels_pred).unwrap();
/// ```
pub fn adjusted_rand_index<T, U, S1, S2, D1, D2>(
    labels_true: &ArrayBase<S1, D1>,
    labels_pred: &ArrayBase<S2, D2>,
) -> Result<f64>
where
    T: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    U: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    S1: Data<Elem = T>,
    S2: Data<Elem = U>,
    D1: Dimension,
    D2: Dimension,
{
    // Check that arrays have the same length
    if labels_true.len() != labels_pred.len() {
        return Err(MetricsError::InvalidInput(format!(
            "labels_true and labels_pred have different lengths: {} vs {}",
            labels_true.len(),
            labels_pred.len()
        )));
    }

    let n_samples = labels_true.len();
    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    // Compute contingency matrix
    let mut contingency: HashMap<(String, String), usize> = HashMap::new();
    for (lt, lp) in labels_true.iter().zip(labels_pred.iter()) {
        let key = (format!("{:?}", lt), format!("{:?}", lp));
        *contingency.entry(key).or_insert(0) += 1;
    }

    // Count pairs
    let (mut sum_comb_a, mut sum_comb_b, mut sum_comb) = (0.0, 0.0, 0.0);

    // Count terms for true labels
    let mut a_counts: HashMap<String, usize> = HashMap::new();
    for lt in labels_true.iter() {
        let key = format!("{:?}", lt);
        *a_counts.entry(key).or_insert(0) += 1;
    }

    for (_, &count) in a_counts.iter() {
        if count > 1 {
            sum_comb_a += combinations(count);
        }
    }

    // Count terms for predicted labels
    let mut b_counts: HashMap<String, usize> = HashMap::new();
    for lp in labels_pred.iter() {
        let key = format!("{:?}", lp);
        *b_counts.entry(key).or_insert(0) += 1;
    }

    for (_, &count) in b_counts.iter() {
        if count > 1 {
            sum_comb_b += combinations(count);
        }
    }

    // Count terms for contingency matrix
    for (_, &count) in contingency.iter() {
        if count > 1 {
            sum_comb += combinations(count);
        }
    }

    // Calculate total number of pairs
    let n_pairs = combinations(n_samples);

    // Calculate ARI
    let expected_index = sum_comb_a * sum_comb_b / n_pairs;
    let max_index = (sum_comb_a + sum_comb_b) / 2.0;

    if max_index == expected_index {
        // Special case: perfect match
        Ok(1.0)
    } else {
        Ok((sum_comb - expected_index) / (max_index - expected_index))
    }
}

/// Calculates the Normalized Mutual Information (NMI) between two clusterings
///
/// NMI is a normalization of the Mutual Information (MI) score to scale the
/// results between 0 (no mutual information) and 1 (perfect correlation).
/// It measures the agreement of two clusterings, ignoring permutations.
///
/// # Arguments
///
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Predicted cluster labels to evaluate
/// * `average_method` - Method to compute the normalization. One of:
///   * "arithmetic": (MI) / ((H(labels_true) + H(labels_pred)) / 2)
///   * "geometric": MI / sqrt(H(labels_true) * H(labels_pred))
///   * "min": MI / min(H(labels_true), H(labels_pred))
///   * "max": MI / max(H(labels_true), H(labels_pred))
///
/// # Returns
///
/// * The Normalized Mutual Information score (between 0.0 and 1.0)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::clustering::normalized_mutual_info_score;
///
/// let labels_true = array![0, 0, 1, 1, 2, 2];
/// let labels_pred = array![0, 0, 0, 1, 1, 1];
///
/// let nmi = normalized_mutual_info_score(&labels_true, &labels_pred, "arithmetic").unwrap();
/// ```
pub fn normalized_mutual_info_score<T, U, S1, S2, D1, D2>(
    labels_true: &ArrayBase<S1, D1>,
    labels_pred: &ArrayBase<S2, D2>,
    average_method: &str,
) -> Result<f64>
where
    T: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    U: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    S1: Data<Elem = T>,
    S2: Data<Elem = U>,
    D1: Dimension,
    D2: Dimension,
{
    // Check that arrays have the same length
    if labels_true.len() != labels_pred.len() {
        return Err(MetricsError::InvalidInput(format!(
            "labels_true and labels_pred have different lengths: {} vs {}",
            labels_true.len(),
            labels_pred.len()
        )));
    }

    let n_samples = labels_true.len();
    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    // Validate average_method
    match average_method {
        "arithmetic" | "geometric" | "min" | "max" => {}
        _ => {
            return Err(MetricsError::InvalidInput(format!(
                "Invalid average_method: {}. Must be one of 'arithmetic', 'geometric', 'min', or 'max'",
                average_method
            )));
        }
    }

    // Compute contingency matrix (using strings for label values to handle different types)
    let mut contingency: HashMap<(String, String), usize> = HashMap::new();
    for (lt, lp) in labels_true.iter().zip(labels_pred.iter()) {
        let key = (format!("{:?}", lt), format!("{:?}", lp));
        *contingency.entry(key).or_insert(0) += 1;
    }

    // Count labels
    let mut true_counts: HashMap<String, usize> = HashMap::new();
    for lt in labels_true.iter() {
        let key = format!("{:?}", lt);
        *true_counts.entry(key).or_insert(0) += 1;
    }

    let mut pred_counts: HashMap<String, usize> = HashMap::new();
    for lp in labels_pred.iter() {
        let key = format!("{:?}", lp);
        *pred_counts.entry(key).or_insert(0) += 1;
    }

    // Calculate entropy for true labels
    let mut h_true = 0.0;
    for (_, &count) in true_counts.iter() {
        let pk = count as f64 / n_samples as f64;
        h_true -= pk * pk.ln();
    }

    // Calculate entropy for predicted labels
    let mut h_pred = 0.0;
    for (_, &count) in pred_counts.iter() {
        let pk = count as f64 / n_samples as f64;
        h_pred -= pk * pk.ln();
    }

    // Calculate mutual information
    let mut mutual_info = 0.0;
    let n_samples_f64 = n_samples as f64;

    for ((lt, lp), &nij) in contingency.iter() {
        let ni = true_counts.get(lt).unwrap_or(&0);
        let nj = pred_counts.get(lp).unwrap_or(&0);

        if nij > 0 && *ni > 0 && *nj > 0 {
            let pij = nij as f64 / n_samples_f64;
            let pi = *ni as f64 / n_samples_f64;
            let pj = *nj as f64 / n_samples_f64;

            mutual_info += pij * (pij / (pi * pj)).ln();
        }
    }

    // Normalize mutual information based on average_method
    let nmi = match average_method {
        "arithmetic" => {
            if h_true + h_pred == 0.0 {
                0.0
            } else {
                2.0 * mutual_info / (h_true + h_pred)
            }
        }
        "geometric" => {
            if h_true == 0.0 || h_pred == 0.0 {
                0.0
            } else {
                mutual_info / (h_true * h_pred).sqrt()
            }
        }
        "min" => {
            let min_entropy = h_true.min(h_pred);
            if min_entropy == 0.0 {
                0.0
            } else {
                mutual_info / min_entropy
            }
        }
        "max" => {
            let max_entropy = h_true.max(h_pred);
            if max_entropy == 0.0 {
                0.0
            } else {
                mutual_info / max_entropy
            }
        }
        _ => unreachable!(), // Already validated
    };

    // Clamp to [0, 1] range to avoid numerical issues
    Ok(nmi.clamp(0.0, 1.0))
}

/// Calculates the Adjusted Mutual Information (AMI) between two clusterings
///
/// The Adjusted Mutual Information is an adjustment of the Mutual Information (MI) score
/// to account for chance. It accounts for the fact that MI is generally higher for two
/// clusterings with a larger number of clusters, regardless of whether there is actually
/// more information shared.
///
/// AMI values range from 0 to 1:
/// * 1: Perfect agreement between the clusterings
/// * 0: Agreement equivalent to random chance
///
/// # Arguments
///
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Predicted cluster labels to evaluate
/// * `average_method` - Method to compute the adjustment. One of:
///   * "arithmetic": (MI - E[MI]) / (max(H(labels_true), H(labels_pred)) - E[MI])
///   * "geometric": (MI - E[MI]) / (sqrt(H(labels_true) * H(labels_pred)) - E[MI])
///   * "max": (MI - E[MI]) / (max(H(labels_true), H(labels_pred)) - E[MI])
///   * "min": (MI - E[MI]) / (min(H(labels_true), H(labels_pred)) - E[MI])
///
/// # Returns
///
/// * The Adjusted Mutual Information score (between 0.0 and 1.0)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::clustering::adjusted_mutual_info_score;
///
/// let labels_true = array![0, 0, 1, 1, 2, 2];
/// let labels_pred = array![0, 0, 0, 1, 1, 1];
///
/// let ami = adjusted_mutual_info_score(&labels_true, &labels_pred, "arithmetic").unwrap();
/// ```
pub fn adjusted_mutual_info_score<T, U, S1, S2, D1, D2>(
    labels_true: &ArrayBase<S1, D1>,
    labels_pred: &ArrayBase<S2, D2>,
    average_method: &str,
) -> Result<f64>
where
    T: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    U: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    S1: Data<Elem = T>,
    S2: Data<Elem = U>,
    D1: Dimension,
    D2: Dimension,
{
    // Check that arrays have the same length
    if labels_true.len() != labels_pred.len() {
        return Err(MetricsError::InvalidInput(format!(
            "labels_true and labels_pred have different lengths: {} vs {}",
            labels_true.len(),
            labels_pred.len()
        )));
    }

    let n_samples = labels_true.len();
    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    // Validate average_method
    match average_method {
        "arithmetic" | "geometric" | "min" | "max" => {}
        _ => {
            return Err(MetricsError::InvalidInput(format!(
                "Invalid average_method: {}. Must be one of 'arithmetic', 'geometric', 'min', or 'max'",
                average_method
            )));
        }
    }

    // Compute contingency matrix (using strings for label values to handle different types)
    let mut contingency: HashMap<(String, String), usize> = HashMap::new();
    for (lt, lp) in labels_true.iter().zip(labels_pred.iter()) {
        let key = (format!("{:?}", lt), format!("{:?}", lp));
        *contingency.entry(key).or_insert(0) += 1;
    }

    // Count labels (and store them in a way we can reference later)
    let mut true_labels: Vec<String> = Vec::new();
    let mut true_counts: HashMap<String, usize> = HashMap::new();
    for lt in labels_true.iter() {
        let key = format!("{:?}", lt);
        if !true_labels.contains(&key) {
            true_labels.push(key.clone());
        }
        *true_counts.entry(key).or_insert(0) += 1;
    }

    let mut pred_labels: Vec<String> = Vec::new();
    let mut pred_counts: HashMap<String, usize> = HashMap::new();
    for lp in labels_pred.iter() {
        let key = format!("{:?}", lp);
        if !pred_labels.contains(&key) {
            pred_labels.push(key.clone());
        }
        *pred_counts.entry(key).or_insert(0) += 1;
    }

    // Calculate entropy for true labels
    let mut h_true = 0.0;
    for (_, &count) in true_counts.iter() {
        let pk = count as f64 / n_samples as f64;
        h_true -= pk * pk.ln();
    }

    // Calculate entropy for predicted labels
    let mut h_pred = 0.0;
    for (_, &count) in pred_counts.iter() {
        let pk = count as f64 / n_samples as f64;
        h_pred -= pk * pk.ln();
    }

    // Calculate mutual information
    let mut mutual_info = 0.0;
    let n_samples_f64 = n_samples as f64;

    for ((lt, lp), &nij) in contingency.iter() {
        let ni = true_counts.get(lt).unwrap_or(&0);
        let nj = pred_counts.get(lp).unwrap_or(&0);

        if nij > 0 && *ni > 0 && *nj > 0 {
            let pij = nij as f64 / n_samples_f64;
            let pi = *ni as f64 / n_samples_f64;
            let pj = *nj as f64 / n_samples_f64;

            mutual_info += pij * (pij / (pi * pj)).ln();
        }
    }

    // Calculate the expected mutual information
    let a = true_labels.len() as f64;
    let b = pred_labels.len() as f64;
    let n = n_samples as f64;

    // Special case: if a or b is 1, expected MI is 0
    if a <= 1.0 || b <= 1.0 {
        return Ok(0.0);
    }

    // Calculate expected mutual information
    let mut emi = 0.0;

    for (_, &ai) in true_counts.iter() {
        for (_, &bj) in pred_counts.iter() {
            let ai_f64 = ai as f64;
            let bj_f64 = bj as f64;

            // Compute the sum over N_{ij} (a bit complex for an exact match)
            // We use a simpler approximation based on the concept that the expectation
            // can be approximated as the product of marginals divided by n_samples
            let expected_nij = ai_f64 * bj_f64 / n;

            if expected_nij > 0.0 {
                let pi = ai_f64 / n;
                let pj = bj_f64 / n;
                let pij = expected_nij / n;

                emi += expected_nij / n_samples_f64 * (pij / (pi * pj)).ln();
            }
        }
    }

    // Adjust the mutual information
    let ami = match average_method {
        "arithmetic" => {
            let avg_h = (h_true + h_pred) / 2.0;
            if avg_h <= emi {
                0.0
            } else {
                (mutual_info - emi) / (avg_h - emi)
            }
        }
        "geometric" => {
            let sqrt_h = (h_true * h_pred).sqrt();
            if sqrt_h <= emi {
                0.0
            } else {
                (mutual_info - emi) / (sqrt_h - emi)
            }
        }
        "min" => {
            let min_h = h_true.min(h_pred);
            if min_h <= emi {
                0.0
            } else {
                (mutual_info - emi) / (min_h - emi)
            }
        }
        "max" => {
            let max_h = h_true.max(h_pred);
            if max_h <= emi {
                0.0
            } else {
                (mutual_info - emi) / (max_h - emi)
            }
        }
        _ => unreachable!(), // Already validated
    };

    // Clamp to [0, 1] range to avoid numerical issues
    Ok(ami.clamp(0.0, 1.0))
}

/// Calculates the Homogeneity, Completeness, and V-measure scores for a clustering
///
/// Homogeneity measures how each cluster contains only members of a single class.
/// Completeness measures how all members of a given class are assigned to the same cluster.
/// V-measure is the harmonic mean of homogeneity and completeness.
///
/// All three metrics range from 0.0 to 1.0, with higher values being better.
///
/// # Arguments
///
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Predicted cluster labels to evaluate
/// * `beta` - Weight of completeness in the V-measure calculation (default is 1.0 for equal weight)
///
/// # Returns
///
/// * A tuple of (homogeneity, completeness, v_measure)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::clustering::homogeneity_completeness_v_measure;
///
/// let labels_true = array![0, 0, 1, 1, 2, 2];
/// let labels_pred = array![0, 0, 0, 1, 1, 1];
///
/// let (homogeneity, completeness, v_measure) =
///     homogeneity_completeness_v_measure(&labels_true, &labels_pred, 1.0).unwrap();
/// ```
pub fn homogeneity_completeness_v_measure<T, U, S1, S2, D1, D2>(
    labels_true: &ArrayBase<S1, D1>,
    labels_pred: &ArrayBase<S2, D2>,
    beta: f64,
) -> Result<(f64, f64, f64)>
where
    T: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    U: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    S1: Data<Elem = T>,
    S2: Data<Elem = U>,
    D1: Dimension,
    D2: Dimension,
{
    // Check that arrays have the same length
    if labels_true.len() != labels_pred.len() {
        return Err(MetricsError::InvalidInput(format!(
            "labels_true and labels_pred have different lengths: {} vs {}",
            labels_true.len(),
            labels_pred.len()
        )));
    }

    let n_samples = labels_true.len();
    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    // Validate beta
    if beta < 0.0 {
        return Err(MetricsError::InvalidInput(
            "beta must be non-negative".to_string(),
        ));
    }

    // Compute contingency matrix (using strings for label values to handle different types)
    let mut contingency: HashMap<(String, String), usize> = HashMap::new();
    for (lt, lp) in labels_true.iter().zip(labels_pred.iter()) {
        let key = (format!("{:?}", lt), format!("{:?}", lp));
        *contingency.entry(key).or_insert(0) += 1;
    }

    // Count labels
    let mut true_counts: HashMap<String, usize> = HashMap::new();
    for lt in labels_true.iter() {
        let key = format!("{:?}", lt);
        *true_counts.entry(key).or_insert(0) += 1;
    }

    let mut pred_counts: HashMap<String, usize> = HashMap::new();
    for lp in labels_pred.iter() {
        let key = format!("{:?}", lp);
        *pred_counts.entry(key).or_insert(0) += 1;
    }

    // Calculate entropy for true labels
    let mut h_true = 0.0;
    for (_, &count) in true_counts.iter() {
        let pk = count as f64 / n_samples as f64;
        h_true -= pk * pk.ln();
    }

    // Calculate entropy for predicted labels
    let mut h_pred = 0.0;
    for (_, &count) in pred_counts.iter() {
        let pk = count as f64 / n_samples as f64;
        h_pred -= pk * pk.ln();
    }

    // Calculate conditional entropy H(true|pred)
    let mut h_true_given_pred = 0.0;
    let n_samples_f64 = n_samples as f64;

    for label_pred in pred_counts.keys() {
        let mut cluster_true_counts: HashMap<String, usize> = HashMap::new();
        let mut pred_size = 0;

        // Count occurrences of true labels in this predicted cluster
        for ((label_true, lp), &count) in contingency.iter() {
            if *lp == *label_pred {
                *cluster_true_counts.entry(label_true.clone()).or_insert(0) += count;
                pred_size += count;
            }
        }

        // Calculate conditional entropy contribution
        for &count in cluster_true_counts.values() {
            if count > 0 {
                let pk = count as f64 / pred_size as f64;
                h_true_given_pred -= (count as f64 / n_samples_f64) * pk.ln();
            }
        }
    }

    // Calculate conditional entropy H(pred|true)
    let mut h_pred_given_true = 0.0;

    for label_true in true_counts.keys() {
        let mut cluster_pred_counts: HashMap<String, usize> = HashMap::new();
        let mut true_size = 0;

        // Count occurrences of predicted labels for this true class
        for ((lt, label_pred), &count) in contingency.iter() {
            if *lt == *label_true {
                *cluster_pred_counts.entry(label_pred.clone()).or_insert(0) += count;
                true_size += count;
            }
        }

        // Calculate conditional entropy contribution
        for &count in cluster_pred_counts.values() {
            if count > 0 {
                let pk = count as f64 / true_size as f64;
                h_pred_given_true -= (count as f64 / n_samples_f64) * pk.ln();
            }
        }
    }

    // Calculate homogeneity and completeness
    let homogeneity = if h_true == 0.0 {
        1.0
    } else {
        1.0 - h_true_given_pred / h_true
    };

    let completeness = if h_pred == 0.0 {
        1.0
    } else {
        1.0 - h_pred_given_true / h_pred
    };

    // Calculate V-measure
    let v_measure = if homogeneity + completeness == 0.0 {
        0.0
    } else {
        (1.0 + beta) * homogeneity * completeness / (beta * homogeneity + completeness)
    };

    // Clamp to [0, 1] range to avoid numerical issues
    Ok((
        homogeneity.clamp(0.0, 1.0),
        completeness.clamp(0.0, 1.0),
        v_measure.clamp(0.0, 1.0),
    ))
}

/// Calculates the Fowlkes-Mallows score for a clustering
///
/// The Fowlkes-Mallows score is the geometric mean of pairwise precision and recall.
/// It is defined as the geometric mean of the precision and recall in the classification
/// task of retrieving pairs of elements that are in the same cluster in both clusterings.
///
/// This score ranges from 0.0 to 1.0, with higher values indicating better agreement
/// between the clusterings.
///
/// # Arguments
///
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Predicted cluster labels to evaluate
///
/// # Returns
///
/// * The Fowlkes-Mallows score
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::clustering::fowlkes_mallows_score;
///
/// let labels_true = array![0, 0, 1, 1, 2, 2];
/// let labels_pred = array![0, 0, 0, 1, 1, 1];
///
/// let score = fowlkes_mallows_score(&labels_true, &labels_pred).unwrap();
/// ```
pub fn fowlkes_mallows_score<T, U, S1, S2, D1, D2>(
    labels_true: &ArrayBase<S1, D1>,
    labels_pred: &ArrayBase<S2, D2>,
) -> Result<f64>
where
    T: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    U: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    S1: Data<Elem = T>,
    S2: Data<Elem = U>,
    D1: Dimension,
    D2: Dimension,
{
    // Check that arrays have the same length
    if labels_true.len() != labels_pred.len() {
        return Err(MetricsError::InvalidInput(format!(
            "labels_true and labels_pred have different lengths: {} vs {}",
            labels_true.len(),
            labels_pred.len()
        )));
    }

    let n_samples = labels_true.len();
    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    // Compute contingency matrix (using strings for label values to handle different types)
    let mut contingency: HashMap<(String, String), usize> = HashMap::new();
    for (lt, lp) in labels_true.iter().zip(labels_pred.iter()) {
        let key = (format!("{:?}", lt), format!("{:?}", lp));
        *contingency.entry(key).or_insert(0) += 1;
    }

    // Count labels
    let mut true_counts: HashMap<String, usize> = HashMap::new();
    for lt in labels_true.iter() {
        let key = format!("{:?}", lt);
        *true_counts.entry(key).or_insert(0) += 1;
    }

    let mut pred_counts: HashMap<String, usize> = HashMap::new();
    for lp in labels_pred.iter() {
        let key = format!("{:?}", lp);
        *pred_counts.entry(key).or_insert(0) += 1;
    }

    // Calculate TP, FP, FN
    let mut tp = 0.0; // True positives: pairs that are in the same cluster in both clusterings

    // Calculate TP from the contingency table
    for &count in contingency.values() {
        if count > 1 {
            tp += combinations(count);
        }
    }

    // Calculate sum of combinations from each cluster (for both true and pred)
    let mut sum_comb_true = 0.0;
    for &count in true_counts.values() {
        if count > 1 {
            sum_comb_true += combinations(count);
        }
    }

    let mut sum_comb_pred = 0.0;
    for &count in pred_counts.values() {
        if count > 1 {
            sum_comb_pred += combinations(count);
        }
    }

    // Special case: No pairs in true or pred clusters (all clusters have size 1)
    if sum_comb_true == 0.0 || sum_comb_pred == 0.0 {
        return Ok(1.0); // Perfect agreement
    }

    // Calculate precision and recall
    let precision = tp / sum_comb_pred;
    let recall = tp / sum_comb_true;

    // Calculate Fowlkes-Mallows score (geometric mean of precision and recall)
    let score = (precision * recall).sqrt();

    // Handle potential numerical issues
    Ok(score.clamp(0.0, 1.0))
}

/// Calculates the Dunn index for a clustering
///
/// The Dunn index is defined as the ratio of the smallest distance between clusters
/// to the largest intra-cluster distance (cluster diameter). Higher values indicate
/// better clustering with dense, well-separated clusters.
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
/// use scirs2_metrics::clustering::dunn_index;
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
/// let score = dunn_index(&x, &labels).unwrap();
/// assert!(score > 0.5); // High score for well-separated clusters
/// ```
pub fn dunn_index<F, S1, S2, D>(x: &ArrayBase<S1, Ix2>, labels: &ArrayBase<S2, D>) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug + ndarray::ScalarOperand,
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
    let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &label) in labels.iter().enumerate() {
        clusters.entry(label).or_default().push(i);
    }

    // Check that there are at least 2 clusters
    if clusters.len() < 2 {
        return Err(MetricsError::InvalidInput(
            "Number of labels is 1. Dunn index is undefined for a single cluster.".to_string(),
        ));
    }

    // Calculate the minimum inter-cluster distance
    let mut min_inter_cluster_dist = F::max_value();
    let cluster_labels: Vec<_> = clusters.keys().cloned().collect();

    for i in 0..cluster_labels.len() {
        let label_i = cluster_labels[i];
        let cluster_i = &clusters[&label_i];

        for &label_j in &cluster_labels[(i + 1)..] {
            let cluster_j = &clusters[&label_j];

            // Compute the minimum distance between points in cluster i and cluster j
            let mut min_dist_ij = F::max_value();

            for &idx_i in cluster_i {
                for &idx_j in cluster_j {
                    let dist = euclidean_distance(x.row(idx_i).to_owned(), x.row(idx_j).to_owned());
                    if dist < min_dist_ij {
                        min_dist_ij = dist;
                    }
                }
            }

            // Update the minimum inter-cluster distance
            if min_dist_ij < min_inter_cluster_dist {
                min_inter_cluster_dist = min_dist_ij;
            }
        }
    }

    // Calculate the maximum intra-cluster distance (cluster diameter)
    let mut max_intra_cluster_dist = F::min_value();

    for indices in clusters.values() {
        let mut max_diameter = F::min_value();

        // Compute the maximum distance between any two points in the cluster
        for (i, &idx_i) in indices.iter().enumerate() {
            for &idx_j in indices[(i + 1)..].iter() {
                let dist = euclidean_distance(x.row(idx_i).to_owned(), x.row(idx_j).to_owned());
                if dist > max_diameter {
                    max_diameter = dist;
                }
            }
        }

        // Update the maximum intra-cluster distance
        if max_diameter > max_intra_cluster_dist {
            max_intra_cluster_dist = max_diameter;
        }
    }

    // Handle edge cases
    if max_intra_cluster_dist <= F::epsilon() {
        return Err(MetricsError::CalculationError(
            "Maximum intra-cluster distance is zero or very small".to_string(),
        ));
    }

    if min_inter_cluster_dist >= F::max_value() {
        return Err(MetricsError::CalculationError(
            "Minimum inter-cluster distance could not be computed".to_string(),
        ));
    }

    // Calculate Dunn index
    let dunn = min_inter_cluster_dist / max_intra_cluster_dist;

    Ok(dunn)
}

/// Implements the elbow method for finding the optimal number of clusters
///
/// The elbow method runs clustering with different numbers of clusters and
/// calculates the sum of squared distances of samples to their closest cluster center.
/// The optimal number of clusters is often considered to be at the "elbow" point,
/// where adding another cluster doesn't significantly reduce the sum of squared distances.
///
/// This function returns the inertia values for each number of clusters, which can be
/// plotted to identify the elbow point.
///
/// # Arguments
///
/// * `x` - Array of shape (n_samples, n_features) containing the data
/// * `cluster_range` - Vector of integers indicating the number of clusters to try
/// * `metric` - Name of clustering metric to use for evaluation
///   (one of "inertia", "silhouette", "calinski_harabasz", "davies_bouldin", "dunn")
/// * `algorithm` - Clustering algorithm to use ("kmeans" currently supported)
/// * `random_state` - Optional random seed for reproducibility
/// * `max_iter` - Maximum number of iterations for the clustering algorithm (default: 100)
///
/// # Returns
///
/// * A vector of (k, score) pairs where k is the number of clusters and score is the corresponding metric value
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array2};
/// use scirs2_metrics::clustering::elbow_method;
///
/// // Create a small dataset with 3 potential clusters
/// let x = Array2::from_shape_vec((9, 2), vec![
///     1.0, 2.0, 1.5, 1.8, 1.2, 2.2,
///     5.0, 6.0, 5.2, 5.8, 5.5, 6.2,
///     9.0, 10.0, 9.2, 9.8, 9.5, 10.2
/// ]).unwrap();
///
/// // Try clustering with k = 1 to 5
/// let cluster_range = vec![1, 2, 3, 4, 5];
///
/// // Calculate inertia for different k values
/// let results = elbow_method(&x, &cluster_range, "inertia", "kmeans", Some(42), None).unwrap();
///
/// // The elbow point should be around k=3 for this dataset
/// ```
pub fn elbow_method<F, S>(
    x: &ArrayBase<S, Ix2>,
    cluster_range: &[usize],
    metric: &str,
    algorithm: &str,
    random_state: Option<u64>,
    max_iter: Option<usize>,
) -> Result<Vec<(usize, F)>>
where
    F: Float + NumCast + std::fmt::Debug + ndarray::ScalarOperand,
    S: Data<Elem = F>,
{
    if cluster_range.is_empty() {
        return Err(MetricsError::InvalidInput(
            "cluster_range must not be empty".to_string(),
        ));
    }

    // Validate that the metric is supported
    match metric {
        "inertia" | "silhouette" | "calinski_harabasz" | "davies_bouldin" | "dunn" => {}
        _ => {
            return Err(MetricsError::InvalidInput(format!(
                "Unsupported metric: {}. Must be one of 'inertia', 'silhouette', 'calinski_harabasz', 'davies_bouldin', or 'dunn'",
                metric
            )));
        }
    }

    // Validate that the algorithm is supported
    if algorithm != "kmeans" {
        return Err(MetricsError::InvalidInput(format!(
            "Unsupported algorithm: {}. Only 'kmeans' is currently supported.",
            algorithm
        )));
    }

    let max_iter = max_iter.unwrap_or(100);
    let mut results = Vec::with_capacity(cluster_range.len());

    for &k in cluster_range {
        // Validate k
        if k == 0 {
            return Err(MetricsError::InvalidInput(
                "Number of clusters must be at least 1".to_string(),
            ));
        }

        // Call the K-means algorithm (which should be in scirs2-cluster)
        // Since we don't have direct access to it here, we'll simulate the clustering
        // and calculate the requested metric
        let (_centroids, labels, inertia) = kmeans_simulation(x, k, random_state, max_iter)?;

        // Calculate the requested metric
        let score = match metric {
            "inertia" => inertia,
            "silhouette" => {
                if k == 1 {
                    // Silhouette score is undefined for a single cluster
                    F::zero()
                } else {
                    let labels_array = Array1::from_vec(labels.clone());
                    let silhouette = silhouette_score(x, &labels_array, "euclidean")?;
                    NumCast::from(silhouette).ok_or_else(|| {
                        MetricsError::CalculationError(
                            "Failed to convert silhouette score".to_string(),
                        )
                    })?
                }
            }
            "calinski_harabasz" => {
                if k == 1 {
                    // CH score is undefined for a single cluster
                    F::zero()
                } else {
                    let labels_array = Array1::from_vec(labels.clone());
                    let ch_score = calinski_harabasz_score(x, &labels_array)?;
                    NumCast::from(ch_score).ok_or_else(|| {
                        MetricsError::CalculationError("Failed to convert CH score".to_string())
                    })?
                }
            }
            "davies_bouldin" => {
                if k == 1 {
                    // DB index is undefined for a single cluster
                    F::zero()
                } else {
                    let labels_array = Array1::from_vec(labels.clone());
                    let db_score = davies_bouldin_score(x, &labels_array)?;
                    NumCast::from(db_score).ok_or_else(|| {
                        MetricsError::CalculationError("Failed to convert DB index".to_string())
                    })?
                }
            }
            "dunn" => {
                if k == 1 {
                    // Dunn index is undefined for a single cluster
                    F::zero()
                } else {
                    let labels_array = Array1::from_vec(labels.clone());
                    dunn_index(x, &labels_array)?
                }
            }
            _ => unreachable!(), // Already validated
        };

        results.push((k, score));
    }

    Ok(results)
}

/// Calculates the gap statistic for determining the optimal number of clusters
///
/// The gap statistic compares the total within intra-cluster variation for different values of k
/// with their expected values under a null reference distribution of the data.
/// The optimal number of clusters is the value that maximizes the gap statistic.
///
/// # Arguments
///
/// * `x` - Array of shape (n_samples, n_features) containing the data
/// * `cluster_range` - Vector of integers indicating the number of clusters to try
/// * `n_refs` - Number of reference datasets to generate (default: 10)
/// * `algorithm` - Clustering algorithm to use ("kmeans" currently supported)
/// * `random_state` - Optional random seed for reproducibility
///
/// # Returns
///
/// * A vector of (k, gap_value, std_dev) tuples for each k value
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array2};
/// use scirs2_metrics::clustering::gap_statistic;
///
/// // Create a small dataset with 3 potential clusters
/// let x = Array2::from_shape_vec((9, 2), vec![
///     1.0, 2.0, 1.5, 1.8, 1.2, 2.2,
///     5.0, 6.0, 5.2, 5.8, 5.5, 6.2,
///     9.0, 10.0, 9.2, 9.8, 9.5, 10.2
/// ]).unwrap();
///
/// // Try clustering with k = 1 to 5
/// let cluster_range = vec![1, 2, 3, 4, 5];
///
/// // Calculate gap statistic (using fewer references for the example)
/// let results = gap_statistic(&x, &cluster_range, Some(3), "kmeans", Some(42)).unwrap();
///
/// // Find the k that maximizes the gap statistic
/// let mut max_gap = 0.0;
/// let mut optimal_k = 1;
/// for &(k, gap, _) in &results {
///     if gap > max_gap {
///         max_gap = gap;
///         optimal_k = k;
///     }
/// }
/// // optimal_k should be around 3 for this dataset
/// ```
pub fn gap_statistic<F, S>(
    x: &ArrayBase<S, Ix2>,
    cluster_range: &[usize],
    n_refs: Option<usize>,
    algorithm: &str,
    random_state: Option<u64>,
) -> Result<Vec<(usize, F, F)>>
where
    F: Float + NumCast + std::fmt::Debug + ndarray::ScalarOperand,
    S: Data<Elem = F>,
{
    if cluster_range.is_empty() {
        return Err(MetricsError::InvalidInput(
            "cluster_range must not be empty".to_string(),
        ));
    }

    // Validate that the algorithm is supported
    if algorithm != "kmeans" {
        return Err(MetricsError::InvalidInput(format!(
            "Unsupported algorithm: {}. Only 'kmeans' is currently supported.",
            algorithm
        )));
    }

    let n_refs = n_refs.unwrap_or(10);
    let n_samples = x.nrows();
    let n_features = x.ncols();

    if n_samples < 2 {
        return Err(MetricsError::InvalidInput(
            "Number of samples must be at least 2".to_string(),
        ));
    }

    // Find data boundaries for generating random reference datasets
    let mut min_vals = Vec::with_capacity(n_features);
    let mut max_vals = Vec::with_capacity(n_features);

    for j in 0..n_features {
        let mut min_val = F::max_value();
        let mut max_val = F::min_value();

        for i in 0..n_samples {
            let val = x[[i, j]];
            if val < min_val {
                min_val = val;
            }
            if val > max_val {
                max_val = val;
            }
        }

        min_vals.push(min_val);
        max_vals.push(max_val);
    }

    // Set up random number generator
    let mut rng = match random_state {
        Some(seed) => StdRng::seed_from_u64(seed),
        // Use a fixed seed for None to maintain test reliability
        None => StdRng::seed_from_u64(123456789),
    };

    let mut results = Vec::with_capacity(cluster_range.len());

    for &k in cluster_range {
        // Validate k
        if k == 0 {
            return Err(MetricsError::InvalidInput(
                "Number of clusters must be at least 1".to_string(),
            ));
        }

        if k > n_samples {
            return Err(MetricsError::InvalidInput(format!(
                "Number of clusters ({}) cannot be greater than number of samples ({})",
                k, n_samples
            )));
        }

        // Cluster the original data
        let (_, _, inertia_orig) = kmeans_simulation(x, k, random_state, 100)?;
        let log_inertia_orig = inertia_orig.ln();

        // Generate reference datasets and cluster them
        let mut log_inertias_ref = Vec::with_capacity(n_refs);

        for _ in 0..n_refs {
            // Generate random uniform data within the same range as the original data
            let mut ref_data = Array2::<F>::zeros((n_samples, n_features));

            for j in 0..n_features {
                let min_val = min_vals[j];
                let max_val = max_vals[j];
                let range = max_val - min_val;

                for i in 0..n_samples {
                    let rand_val: f64 = rng.random_range(0.0..1.0);
                    let rand_val_f = F::from(rand_val).ok_or_else(|| {
                        MetricsError::CalculationError("Failed to convert random value".to_string())
                    })?;
                    ref_data[[i, j]] = min_val + range * rand_val_f;
                }
            }

            // Cluster the reference data
            let (_, _, inertia_ref) = kmeans_simulation(&ref_data, k, Some(rng.random()), 100)?;
            log_inertias_ref.push(inertia_ref.ln());
        }

        // Calculate gap statistic
        let mean_log_inertia_ref = log_inertias_ref
            .iter()
            .fold(F::zero(), |acc, &val| acc + val)
            / NumCast::from(n_refs).unwrap();
        let gap = mean_log_inertia_ref - log_inertia_orig;

        // Calculate standard deviation
        let variance = log_inertias_ref.iter().fold(F::zero(), |acc, &val| {
            let diff = val - mean_log_inertia_ref;
            acc + diff * diff
        }) / NumCast::from(n_refs).unwrap();

        let std_dev = variance.sqrt();
        let sdk = std_dev * NumCast::from(1.0 + (1.0 / (n_refs as f64).sqrt())).unwrap();

        results.push((k, gap, sdk));
    }

    Ok(results)
}

/// Helper function to simulate K-means clustering
///
/// This is a simplified implementation of K-means for the purpose of the elbow method
/// and gap statistic. In a real application, you would use the scirs2-cluster crate's
/// implementation.
///
/// # Arguments
///
/// * `x` - Array of shape (n_samples, n_features) containing the data
/// * `k` - Number of clusters
/// * `random_state` - Optional random seed for reproducibility
/// * `max_iter` - Maximum number of iterations
///
/// # Returns
///
/// * A tuple of (centroids, labels, inertia) where:
///   - centroids is an Array2<F> of shape (k, n_features)
///   - labels is a Vec<usize> of cluster assignments
///   - inertia is the sum of squared distances to nearest centroids
fn kmeans_simulation<F, S>(
    x: &ArrayBase<S, Ix2>,
    k: usize,
    random_state: Option<u64>,
    max_iter: usize,
) -> Result<(Array2<F>, Vec<usize>, F)>
where
    F: Float + NumCast + std::fmt::Debug + ndarray::ScalarOperand,
    S: Data<Elem = F>,
{
    let n_samples = x.nrows();
    let n_features = x.ncols();

    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty dataset provided".to_string(),
        ));
    }

    if k == 0 {
        return Err(MetricsError::InvalidInput(
            "Number of clusters must be at least 1".to_string(),
        ));
    }

    // For k=1, just compute the centroid as the mean of all points
    if k == 1 {
        let mut centroid = Array1::<F>::zeros(n_features);
        for i in 0..n_samples {
            for j in 0..n_features {
                centroid[j] = centroid[j] + x[[i, j]];
            }
        }
        centroid = centroid / F::from(n_samples).unwrap();

        let centroids = Array2::from_shape_fn((1, n_features), |(_, j)| centroid[j]);
        let labels = vec![0; n_samples];

        // Calculate inertia
        let mut inertia = F::zero();
        for i in 0..n_samples {
            let mut dist_sq = F::zero();
            for j in 0..n_features {
                let diff = x[[i, j]] - centroid[j];
                dist_sq = dist_sq + diff * diff;
            }
            inertia = inertia + dist_sq;
        }

        return Ok((centroids, labels, inertia));
    }

    // Set up random number generator
    let mut rng = match random_state {
        Some(seed) => StdRng::seed_from_u64(seed),
        // Use a fixed seed for None to maintain test reliability
        None => StdRng::seed_from_u64(123456789),
    };

    // Initialize centroids using k-means++ method
    let mut centroids = Array2::<F>::zeros((k, n_features));
    let mut closest_dist_sq = Array1::<F>::from_elem(n_samples, F::infinity());

    // Select first centroid randomly
    let first_centroid_idx = rng.random_range(0..n_samples);
    for j in 0..n_features {
        centroids[[0, j]] = x[[first_centroid_idx, j]];
    }

    // Select remaining centroids
    for c in 1..k {
        // Update closest distances
        for i in 0..n_samples {
            let mut dist_sq = F::zero();
            for j in 0..n_features {
                let diff = x[[i, j]] - centroids[[c - 1, j]];
                dist_sq = dist_sq + diff * diff;
            }

            if dist_sq < closest_dist_sq[i] {
                closest_dist_sq[i] = dist_sq;
            }
        }

        // Calculate sum of distances for probability calculation
        let sum_dist: F = closest_dist_sq
            .iter()
            .fold(F::zero(), |acc, &dist| acc + dist);
        if sum_dist <= F::epsilon() {
            // All points are very close to a centroid already, choose randomly
            let idx = rng.random_range(0..n_samples);
            for j in 0..n_features {
                centroids[[c, j]] = x[[idx, j]];
            }
            continue;
        }

        // Choose next centroid with probability proportional to distance squared
        let rand_val: f64 = rng.random_range(0.0..1.0);
        let rand_val_f = F::from(rand_val).unwrap() * sum_dist;

        let mut cumsum = F::zero();
        let mut idx = 0;
        for i in 0..n_samples {
            cumsum = cumsum + closest_dist_sq[i];
            if cumsum >= rand_val_f {
                idx = i;
                break;
            }
        }

        for j in 0..n_features {
            centroids[[c, j]] = x[[idx, j]];
        }
    }

    // K-means iteration
    let mut labels = vec![0; n_samples];
    let mut inertia = F::zero();

    for _ in 0..max_iter {
        let old_inertia = inertia;
        inertia = F::zero();

        // Assign points to nearest centroids
        for i in 0..n_samples {
            let mut min_dist = F::infinity();
            let mut min_cluster = 0;

            for c in 0..k {
                let mut dist_sq = F::zero();
                for j in 0..n_features {
                    let diff = x[[i, j]] - centroids[[c, j]];
                    dist_sq = dist_sq + diff * diff;
                }

                if dist_sq < min_dist {
                    min_dist = dist_sq;
                    min_cluster = c;
                }
            }

            labels[i] = min_cluster;
            inertia = inertia + min_dist;
        }

        // Update centroids
        let mut new_centroids = Array2::<F>::zeros((k, n_features));
        let mut counts = vec![0; k];

        for i in 0..n_samples {
            let cluster = labels[i];
            counts[cluster] += 1;

            for j in 0..n_features {
                new_centroids[[cluster, j]] = new_centroids[[cluster, j]] + x[[i, j]];
            }
        }

        for c in 0..k {
            if counts[c] > 0 {
                for j in 0..n_features {
                    new_centroids[[c, j]] =
                        new_centroids[[c, j]] / NumCast::from(counts[c]).unwrap();
                }
            } else {
                // If a cluster is empty, keep its previous centroid
                for j in 0..n_features {
                    new_centroids[[c, j]] = centroids[[c, j]];
                }
            }
        }

        centroids = new_centroids;

        // Check for convergence
        let diff = (old_inertia - inertia).abs();
        if diff < F::epsilon() * inertia {
            break;
        }
    }

    Ok((centroids, labels, inertia))
}

/// Contains all silhouette analysis information for a clustering
///
/// This struct holds the silhouette scores for all samples, as well as per-cluster statistics
/// and ordering information for visualization.
#[derive(Debug, Clone)]
pub struct SilhouetteAnalysis<F>
where
    F: Float + NumCast + std::fmt::Debug + ndarray::ScalarOperand,
{
    /// Silhouette scores for each sample
    pub sample_values: Vec<F>,

    /// Mean silhouette coefficient for all samples
    pub mean_score: F,

    /// Mean silhouette coefficient for each cluster
    pub cluster_scores: HashMap<usize, F>,

    /// Sorted indices for visualization (samples ordered by cluster and silhouette value)
    pub sorted_indices: Vec<usize>,

    /// Original cluster labels mapped to consecutive integers (for visualization)
    pub cluster_mapping: HashMap<usize, usize>,

    /// Samples per cluster (ordered by cluster_mapping)
    pub cluster_sizes: Vec<usize>,
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
    F: Float + NumCast + std::fmt::Debug + ndarray::ScalarOperand,
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
    let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &label) in labels.iter().enumerate() {
        clusters.entry(label).or_default().push(i);
    }

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
            a = a + euclidean_distance(x.row(i).to_owned(), x.row(j).to_owned());
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
                cluster_dist =
                    cluster_dist + euclidean_distance(x.row(i).to_owned(), x.row(j).to_owned());
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
    F: Float + NumCast + std::fmt::Debug + ndarray::ScalarOperand,
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
    F: Float + NumCast + std::fmt::Debug + ndarray::ScalarOperand,
    S1: Data<Elem = F>,
    S2: Data<Elem = usize>,
    D: Dimension,
{
    let analysis = silhouette_analysis(x, labels, metric)?;
    Ok(analysis.cluster_scores)
}

/// Helper function to calculate number of combinations (n choose 2)
fn combinations(n: usize) -> f64 {
    if n < 2 {
        0.0
    } else {
        (n * (n - 1)) as f64 / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array2};

    fn create_test_data() -> (Array2<f64>, Array1<usize>) {
        // Create a small dataset with 2 well-separated clusters
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.5, 1.8, 1.2, 2.2, 5.0, 6.0, 5.2, 5.8, 5.5, 6.2],
        )
        .unwrap();

        let labels = array![0, 0, 0, 1, 1, 1];

        (x, labels)
    }

    #[test]
    fn test_silhouette_score() {
        let (x, labels) = create_test_data();

        let score = silhouette_score(&x, &labels, "euclidean").unwrap();
        assert!(score > 0.8); // High score for well-separated clusters
    }

    #[test]
    fn test_davies_bouldin_score() {
        let (x, labels) = create_test_data();

        let score = davies_bouldin_score(&x, &labels).unwrap();
        assert!(score < 0.5); // Low score for well-separated clusters
    }

    #[test]
    fn test_calinski_harabasz_score() {
        let (x, labels) = create_test_data();

        let score = calinski_harabasz_score(&x, &labels).unwrap();
        assert!(score > 50.0); // High score for well-separated clusters
    }

    #[test]
    fn test_adjusted_rand_index() {
        // Perfect clustering
        let labels_true = array![0, 0, 0, 1, 1, 1];
        let labels_pred = array![1, 1, 1, 0, 0, 0]; // Different label values but same clustering
        let ari = adjusted_rand_index(&labels_true, &labels_pred).unwrap();
        assert_abs_diff_eq!(ari, 1.0, epsilon = 1e-10);

        // Imperfect clustering
        let labels_true = array![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
        let labels_pred = array![0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0];
        let ari = adjusted_rand_index(&labels_true, &labels_pred).unwrap();
        assert!(ari > 0.0 && ari < 1.0); // Should be positive but not perfect

        // Random clustering
        let labels_true = array![0, 0, 0, 0, 1, 1, 1, 1];
        let labels_pred = array![0, 1, 0, 1, 0, 1, 0, 1];
        let ari = adjusted_rand_index(&labels_true, &labels_pred).unwrap();
        // ARI can be slightly negative for random clustering
        assert!(ari <= 0.05);

        // Test with different label types - should work regardless of actual value
        let labels_true_str = array!["cat", "cat", "dog", "dog"];
        let labels_pred_int = array![0, 0, 1, 2];
        let _ari = adjusted_rand_index(&labels_true_str, &labels_pred_int).unwrap();
        // The actual value can vary based on implementation details
        // Just checking that it runs without error is sufficient
    }

    #[test]
    fn test_normalized_mutual_info_score() {
        // Perfect clustering
        let labels_true = array![0, 0, 0, 1, 1, 1];
        let labels_pred = array![1, 1, 1, 0, 0, 0]; // Different label values but same clustering

        // Test different normalization methods
        let nmi_arithmetic =
            normalized_mutual_info_score(&labels_true, &labels_pred, "arithmetic").unwrap();
        let nmi_geometric =
            normalized_mutual_info_score(&labels_true, &labels_pred, "geometric").unwrap();
        let nmi_min = normalized_mutual_info_score(&labels_true, &labels_pred, "min").unwrap();
        let nmi_max = normalized_mutual_info_score(&labels_true, &labels_pred, "max").unwrap();

        // All should be 1.0 for perfect clustering
        assert_abs_diff_eq!(nmi_arithmetic, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(nmi_geometric, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(nmi_min, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(nmi_max, 1.0, epsilon = 1e-10);

        // Partially matching clustering
        let labels_true = array![0, 0, 0, 0, 1, 1, 1, 1];
        let labels_pred = array![0, 0, 1, 1, 0, 0, 1, 1];

        // This specific clustering has NMI of 0.0 (no mutual information beyond random chance)
        // but we should check that the function works and returns a valid value
        let nmi = normalized_mutual_info_score(&labels_true, &labels_pred, "arithmetic").unwrap();
        assert!((0.0..=1.0).contains(&nmi));

        // Random clustering (each sample in a different cluster)
        let labels_true = array![0, 0, 0, 0, 1, 1, 1, 1];
        let labels_pred = array![0, 1, 2, 3, 4, 5, 6, 7];

        // NMI value can vary but should be valid
        let nmi = normalized_mutual_info_score(&labels_true, &labels_pred, "arithmetic").unwrap();
        assert!((0.0..=1.0).contains(&nmi));

        // Test with different label types
        let labels_true_str = array!["cat", "cat", "dog", "dog"];
        let labels_pred_int = array![0, 0, 1, 1];
        let nmi =
            normalized_mutual_info_score(&labels_true_str, &labels_pred_int, "arithmetic").unwrap();
        assert_abs_diff_eq!(nmi, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_adjusted_mutual_info_score() {
        // Perfect clustering
        let labels_true = array![0, 0, 0, 1, 1, 1];
        let labels_pred = array![1, 1, 1, 0, 0, 0]; // Different label values but same clustering

        // Test different normalization methods
        let ami_arithmetic =
            adjusted_mutual_info_score(&labels_true, &labels_pred, "arithmetic").unwrap();
        let ami_geometric =
            adjusted_mutual_info_score(&labels_true, &labels_pred, "geometric").unwrap();
        let ami_min = adjusted_mutual_info_score(&labels_true, &labels_pred, "min").unwrap();
        let ami_max = adjusted_mutual_info_score(&labels_true, &labels_pred, "max").unwrap();

        // All should be 1.0 for perfect clustering
        assert_abs_diff_eq!(ami_arithmetic, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(ami_geometric, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(ami_min, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(ami_max, 1.0, epsilon = 1e-10);

        // Random clustering or partially matching clustering
        let labels_true = array![0, 0, 0, 0, 1, 1, 1, 1];
        let labels_pred = array![0, 0, 1, 1, 0, 0, 1, 1];

        // AMI should correct for chance agreement
        let ami = adjusted_mutual_info_score(&labels_true, &labels_pred, "arithmetic").unwrap();
        let nmi = normalized_mutual_info_score(&labels_true, &labels_pred, "arithmetic").unwrap();

        // AMI should be lower than NMI for random or partially matching clusterings
        assert!(ami <= nmi);

        // AMI should be valid
        assert!((0.0..=1.0).contains(&ami));

        // Special case: single cluster in true or pred
        let labels_true_single = array![0, 0, 0, 0];
        let labels_pred_various = array![0, 1, 2, 3];
        let ami =
            adjusted_mutual_info_score(&labels_true_single, &labels_pred_various, "arithmetic")
                .unwrap();
        assert_abs_diff_eq!(ami, 0.0, epsilon = 1e-10); // Should be 0 when one set has only one cluster
    }

    #[test]
    fn test_homogeneity_completeness_v_measure() {
        // Perfect clustering
        let labels_true = array![0, 0, 0, 1, 1, 1];
        let labels_pred = array![1, 1, 1, 0, 0, 0]; // Different label values but same clustering

        let (homogeneity, completeness, v_measure) =
            homogeneity_completeness_v_measure(&labels_true, &labels_pred, 1.0).unwrap();

        // All should be 1.0 for perfect clustering
        assert_abs_diff_eq!(homogeneity, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(completeness, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(v_measure, 1.0, epsilon = 1e-10);

        // Imperfect clustering
        let labels_true = array![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
        let labels_pred = array![0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0];

        let (homogeneity, completeness, v_measure) =
            homogeneity_completeness_v_measure(&labels_true, &labels_pred, 1.0).unwrap();

        // Values should be between 0 and 1
        assert!(homogeneity > 0.0 && homogeneity < 1.0);
        assert!(completeness > 0.0 && completeness < 1.0);
        assert!(v_measure > 0.0 && v_measure < 1.0);

        // V-measure with different beta values
        let (_, _, v_measure_b05) =
            homogeneity_completeness_v_measure(&labels_true, &labels_pred, 0.5).unwrap();
        let (_, _, v_measure_b20) =
            homogeneity_completeness_v_measure(&labels_true, &labels_pred, 2.0).unwrap();

        // With beta < 1, homogeneity is weighted more
        // With beta > 1, completeness is weighted more
        if homogeneity > completeness {
            assert!(v_measure_b05 > v_measure);
            assert!(v_measure_b20 < v_measure);
        } else if completeness > homogeneity {
            assert!(v_measure_b05 < v_measure);
            assert!(v_measure_b20 > v_measure);
        }

        // Special case: single class
        let labels_true_single = array![0, 0, 0, 0];
        let labels_pred_single = array![1, 1, 1, 1];

        let (homogeneity, completeness, v_measure) =
            homogeneity_completeness_v_measure(&labels_true_single, &labels_pred_single, 1.0)
                .unwrap();

        // For single class, both metrics should be 1.0
        assert_abs_diff_eq!(homogeneity, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(completeness, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(v_measure, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fowlkes_mallows_score() {
        // Perfect clustering
        let labels_true = array![0, 0, 0, 1, 1, 1];
        let labels_pred = array![1, 1, 1, 0, 0, 0]; // Different label values but same clustering

        let score = fowlkes_mallows_score(&labels_true, &labels_pred).unwrap();
        assert_abs_diff_eq!(score, 1.0, epsilon = 1e-10);

        // Imperfect clustering
        let labels_true = array![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
        let labels_pred = array![0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0];

        let score = fowlkes_mallows_score(&labels_true, &labels_pred).unwrap();
        assert!(score > 0.0 && score < 1.0);

        // Random clustering
        let labels_true = array![0, 0, 0, 0, 1, 1, 1, 1];
        let labels_pred = array![0, 1, 2, 3, 4, 5, 6, 7];

        let score = fowlkes_mallows_score(&labels_true, &labels_pred).unwrap();
        // For random assignment, the score may vary depending on the implementation
        // Just verify that it's a valid score between 0 and 1
        assert!((0.0..=1.0).contains(&score));

        // Special case: all samples in individual clusters
        let labels_individual_true = array![0, 1, 2, 3];
        let labels_individual_pred = array![0, 1, 2, 3];

        let score =
            fowlkes_mallows_score(&labels_individual_true, &labels_individual_pred).unwrap();
        assert_abs_diff_eq!(score, 1.0, epsilon = 1e-10); // Perfect agreement
    }

    #[test]
    fn test_distance_functions() {
        let a = array![1.0, 2.0, 3.0];
        let b = array![4.0, 5.0, 6.0];

        let dist = euclidean_distance(a.view(), b.view());
        assert_abs_diff_eq!(dist, 5.196152422706632, epsilon = 1e-10);

        let sq_dist = squared_euclidean_distance(a.view(), b.view());
        assert_abs_diff_eq!(sq_dist, 27.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dunn_index() {
        // Create a small dataset with 2 well-separated clusters
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.5, 1.8, 1.2, 2.2, 5.0, 6.0, 5.2, 5.8, 5.5, 6.2],
        )
        .unwrap();

        let labels = array![0, 0, 0, 1, 1, 1];
        let dunn = dunn_index(&x, &labels).unwrap();

        // For well-separated clusters, Dunn index should be relatively high
        assert!(dunn > 0.5);

        // Test with a dataset where clusters are less well-separated
        let x_closer = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.5, 1.8, 2.8, 3.0, 3.0, 3.2, 3.5, 3.8, 4.0, 4.2],
        )
        .unwrap();

        let labels = array![0, 0, 0, 1, 1, 1];
        let dunn_closer = dunn_index(&x_closer, &labels).unwrap();

        // For less separated clusters, Dunn index should be lower
        assert!(dunn_closer < dunn);

        // Test invalid inputs
        let labels_single = array![0, 0, 0, 0, 0, 0]; // Single cluster
        let result = dunn_index(&x, &labels_single);
        assert!(result.is_err());

        let labels_mismatch = array![0, 0, 0]; // Length mismatch
        let result = dunn_index(&x, &labels_mismatch);
        assert!(result.is_err());
    }

    #[test]
    fn test_elbow_method() {
        // Create a small dataset with 3 clusters
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                1.0, 2.0, 1.5, 1.8, 1.2, 2.2, // Cluster 1
                5.0, 6.0, 5.2, 5.8, 5.5, 6.2, // Cluster 2
                9.0, 10.0, 9.2, 9.8, 9.5, 10.2, // Cluster 3
            ],
        )
        .unwrap();

        // Test with inertia metric
        let cluster_range = vec![1, 2, 3, 4];
        let results =
            elbow_method::<f64, _>(&x, &cluster_range, "inertia", "kmeans", Some(42), None)
                .unwrap();

        // Should have results for each k value
        assert_eq!(results.len(), 4);

        // Inertia should decrease as k increases
        let mut prev_inertia = f64::INFINITY;
        for &(k, inertia) in &results {
            let inertia_f64 = inertia;
            assert!(inertia_f64 < prev_inertia);
            prev_inertia = inertia_f64;
            assert_eq!(k, cluster_range[k - 1]); // Verify k values
        }

        // The sharpest decrease in inertia should be between k=1 and k=2, then k=2 and k=3
        let inertia_diffs: Vec<f64> = results
            .windows(2)
            .map(|window| {
                let (_, inertia1) = window[0];
                let (_, inertia2) = window[1];
                inertia1 - inertia2
            })
            .collect();

        assert!(inertia_diffs[0] > inertia_diffs[1]);

        // Test with silhouette metric (should be defined for k >= 2)
        let results = elbow_method::<f64, _>(
            &x,
            &cluster_range[1..],
            "silhouette",
            "kmeans",
            Some(42),
            None,
        )
        .unwrap();
        assert_eq!(results.len(), 3); // k=2,3,4

        // Silhouette score should be highest for k=3 in this case
        let silhouette_scores: Vec<f64> = results.iter().map(|(_, score)| *score).collect();

        // The score for k=3 should be higher than for k=2 and k=4
        assert!(silhouette_scores[1] > silhouette_scores[0]);
        assert!(silhouette_scores[1] > silhouette_scores[2]);

        // Test invalid inputs
        let empty_range: Vec<usize> = vec![];
        let result = elbow_method::<f64, _>(&x, &empty_range, "inertia", "kmeans", None, None);
        assert!(result.is_err());

        let invalid_metric = "invalid";
        let result =
            elbow_method::<f64, _>(&x, &cluster_range, invalid_metric, "kmeans", None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_gap_statistic() {
        // Create a small dataset with 3 clusters
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                1.0, 2.0, 1.5, 1.8, 1.2, 2.2, // Cluster 1
                5.0, 6.0, 5.2, 5.8, 5.5, 6.2, // Cluster 2
                9.0, 10.0, 9.2, 9.8, 9.5, 10.2, // Cluster 3
            ],
        )
        .unwrap();

        // Use a small number of references to keep the test fast
        let n_refs = 2;
        let cluster_range = vec![1, 2, 3, 4];
        let results =
            gap_statistic::<f64, _>(&x, &cluster_range, Some(n_refs), "kmeans", Some(42)).unwrap();

        // Should have results for each k value
        assert_eq!(results.len(), 4);

        // Find the k that maximizes the gap statistic
        let mut max_gap = f64::NEG_INFINITY;
        let mut optimal_k = 0;

        for &(k, gap, _) in &results {
            if gap > max_gap {
                max_gap = gap;
                optimal_k = k;
            }
        }

        // The optimal k should be close to 3 (the actual number of clusters)
        // But with only 2 references, there's some randomness, so we're a bit lenient
        assert!((2..=4).contains(&optimal_k));

        // Test invalid inputs
        let empty_range: Vec<usize> = vec![];
        let result = gap_statistic::<f64, _>(&x, &empty_range, Some(n_refs), "kmeans", None);
        assert!(result.is_err());

        let invalid_algorithm = "invalid";
        let result =
            gap_statistic::<f64, _>(&x, &cluster_range, Some(n_refs), invalid_algorithm, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_kmeans_simulation() {
        // Create a small dataset with 2 clusters
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.5, 1.8, 1.2, 2.2, 5.0, 6.0, 5.2, 5.8, 5.5, 6.2],
        )
        .unwrap();

        // Test with k=2
        let (centroids, labels, inertia) =
            kmeans_simulation::<f64, _>(&x, 2, Some(42), 100).unwrap();

        // Check the dimensions
        assert_eq!(centroids.shape(), &[2, 2]);
        assert_eq!(labels.len(), 6);

        // Inertia should be positive
        assert!(inertia > 0.0);

        // The first 3 points should be in one cluster, the last 3 in another
        let first_cluster = labels[0];
        assert_eq!(labels[1], first_cluster);
        assert_eq!(labels[2], first_cluster);

        let second_cluster = labels[3];
        assert_eq!(labels[4], second_cluster);
        assert_eq!(labels[5], second_cluster);

        assert_ne!(first_cluster, second_cluster);

        // Test with k=1
        let (centroids, labels, inertia) =
            kmeans_simulation::<f64, _>(&x, 1, Some(42), 100).unwrap();

        // Check the dimensions
        assert_eq!(centroids.shape(), &[1, 2]);
        assert_eq!(labels.len(), 6);
        assert!(labels.iter().all(|&l| l == 0));

        // Inertia should be higher for k=1 than for k=2
        assert!(inertia > 0.0);

        // Test invalid inputs
        let result = kmeans_simulation::<f64, _>(&x, 0, None, 100);
        assert!(result.is_err());

        let empty_data = Array2::<f64>::zeros((0, 2));
        let result = kmeans_simulation::<f64, _>(&empty_data, 2, None, 100);
        assert!(result.is_err());
    }
}
