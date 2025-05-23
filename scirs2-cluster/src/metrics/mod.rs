//! Clustering evaluation metrics
//!
//! This module provides metrics for evaluating clustering algorithms performance:
//! - Silhouette coefficient for measuring cluster cohesion and separation
//! - Davies-Bouldin index for evaluating cluster separation
//! - Calinski-Harabasz index for measuring between-cluster vs within-cluster variance

mod silhouette;
pub use silhouette::{silhouette_samples, silhouette_score};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
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

/// Adjusted Rand Index for comparing two clusterings.
///
/// The Adjusted Rand Index (ARI) is a measure of the similarity between two data clusterings,
/// adjusted for chance. It has a value between -1 and 1, where:
/// - 1 indicates perfect agreement
/// - 0 indicates agreement no better than random chance
/// - Negative values indicate agreement worse than random chance
///
/// # Arguments
///
/// * `labels_true` - Ground truth cluster labels
/// * `labels_pred` - Predicted cluster labels
///
/// # Returns
///
/// The Adjusted Rand Index score
///
/// # Example
///
/// ```
/// use ndarray::Array1;
/// use scirs2_cluster::metrics::adjusted_rand_index;
///
/// let labels_true = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
/// let labels_pred = Array1::from_vec(vec![0, 0, 2, 2, 1, 1]);
///
/// let ari: f64 = adjusted_rand_index(labels_true.view(), labels_pred.view()).unwrap();
/// assert!(ari > 0.0);  // Should be positive for similar clusterings
/// ```
pub fn adjusted_rand_index<F>(
    labels_true: ArrayView1<i32>,
    labels_pred: ArrayView1<i32>,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    if labels_true.len() != labels_pred.len() {
        return Err(ClusteringError::InvalidInput(
            "Labels arrays must have the same length".to_string(),
        ));
    }

    let n = labels_true.len();
    if n == 0 {
        return Err(ClusteringError::InvalidInput(
            "Empty labels arrays".to_string(),
        ));
    }

    // Build contingency table
    let mut true_labels = std::collections::HashSet::new();
    let mut pred_labels = std::collections::HashSet::new();

    for &label in labels_true.iter() {
        true_labels.insert(label);
    }
    for &label in labels_pred.iter() {
        pred_labels.insert(label);
    }

    let n_true = true_labels.len();
    let n_pred = pred_labels.len();

    // Create mapping from labels to indices
    let true_label_map: std::collections::HashMap<_, _> = true_labels
        .iter()
        .enumerate()
        .map(|(i, &label)| (label, i))
        .collect();
    let pred_label_map: std::collections::HashMap<_, _> = pred_labels
        .iter()
        .enumerate()
        .map(|(i, &label)| (label, i))
        .collect();

    // Build contingency table
    let mut contingency = Array2::<usize>::zeros((n_true, n_pred));
    for i in 0..n {
        let true_idx = true_label_map[&labels_true[i]];
        let pred_idx = pred_label_map[&labels_pred[i]];
        contingency[[true_idx, pred_idx]] += 1;
    }

    // Calculate sums
    let sum_comb_c = contingency
        .iter()
        .map(|&n_ij| {
            if n_ij >= 2 {
                (n_ij * (n_ij - 1)) / 2
            } else {
                0
            }
        })
        .sum::<usize>();

    let sum_a = contingency
        .sum_axis(Axis(1))
        .iter()
        .map(|&n_i| if n_i >= 2 { (n_i * (n_i - 1)) / 2 } else { 0 })
        .sum::<usize>();

    let sum_b = contingency
        .sum_axis(Axis(0))
        .iter()
        .map(|&n_j| if n_j >= 2 { (n_j * (n_j - 1)) / 2 } else { 0 })
        .sum::<usize>();

    let n_choose_2 = if n >= 2 { (n * (n - 1)) / 2 } else { 0 };

    // Calculate expected index
    let expected_index =
        F::from(sum_a).unwrap() * F::from(sum_b).unwrap() / F::from(n_choose_2).unwrap();
    let max_index = (F::from(sum_a).unwrap() + F::from(sum_b).unwrap()) / F::from(2.0).unwrap();
    let index = F::from(sum_comb_c).unwrap();

    // Handle edge cases
    if max_index == expected_index {
        return Ok(F::zero());
    }

    // Calculate ARI
    let ari = (index - expected_index) / (max_index - expected_index);
    Ok(ari)
}

/// Normalized Mutual Information (NMI) for comparing two clusterings.
///
/// Normalized Mutual Information is a normalization of the Mutual Information (MI) score
/// to scale the results between 0 (no mutual information) and 1 (perfect correlation).
///
/// # Arguments
///
/// * `labels_true` - Ground truth cluster labels
/// * `labels_pred` - Predicted cluster labels
/// * `average_method` - Method to compute the normalizer ('geometric', 'arithmetic', 'min', 'max')
///
/// # Returns
///
/// The Normalized Mutual Information score
///
/// # Example
///
/// ```
/// use ndarray::Array1;
/// use scirs2_cluster::metrics::normalized_mutual_info;
///
/// let labels_true = Array1::from_vec(vec![0, 0, 1, 1]);
/// let labels_pred = Array1::from_vec(vec![0, 0, 1, 1]);
///
/// let nmi: f64 = normalized_mutual_info(labels_true.view(), labels_pred.view(), "arithmetic").unwrap();
/// assert!((nmi - 1.0).abs() < 1e-6);  // Perfect agreement
/// ```
pub fn normalized_mutual_info<F>(
    labels_true: ArrayView1<i32>,
    labels_pred: ArrayView1<i32>,
    average_method: &str,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    if labels_true.len() != labels_pred.len() {
        return Err(ClusteringError::InvalidInput(
            "Labels arrays must have the same length".to_string(),
        ));
    }

    let n = labels_true.len();
    if n == 0 {
        return Ok(F::one());
    }

    // Compute mutual information
    let mi = mutual_info::<F>(labels_true, labels_pred)?;

    // Compute entropies
    let h_true = entropy::<F>(labels_true)?;
    let h_pred = entropy::<F>(labels_pred)?;

    // Handle edge cases
    if h_true == F::zero() && h_pred == F::zero() {
        return Ok(F::one());
    }

    // Compute normalization
    let normalizer = match average_method {
        "arithmetic" => (h_true + h_pred) / F::from(2.0).unwrap(),
        "geometric" => (h_true * h_pred).sqrt(),
        "min" => h_true.min(h_pred),
        "max" => h_true.max(h_pred),
        _ => {
            return Err(ClusteringError::InvalidInput(
                "Invalid average method. Use 'arithmetic', 'geometric', 'min', or 'max'"
                    .to_string(),
            ))
        }
    };

    if normalizer == F::zero() {
        return Ok(F::zero());
    }

    Ok(mi / normalizer)
}

/// Compute mutual information between two label assignments.
fn mutual_info<F>(labels_true: ArrayView1<i32>, labels_pred: ArrayView1<i32>) -> Result<F>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let n = labels_true.len() as f64;
    let contingency = build_contingency_matrix(labels_true, labels_pred)?;

    let mut mi = F::zero();
    let n_rows = contingency.shape()[0];
    let n_cols = contingency.shape()[1];

    // Compute marginal sums
    let row_sums = contingency.sum_axis(Axis(1));
    let col_sums = contingency.sum_axis(Axis(0));

    for i in 0..n_rows {
        for j in 0..n_cols {
            let n_ij = contingency[[i, j]] as f64;
            if n_ij > 0.0 {
                let n_i = row_sums[i] as f64;
                let n_j = col_sums[j] as f64;
                let term = n_ij / n * (n_ij / (n_i * n_j / n)).ln();
                mi = mi + F::from(term).unwrap();
            }
        }
    }

    Ok(mi)
}

/// Compute entropy of a label assignment.
fn entropy<F>(labels: ArrayView1<i32>) -> Result<F>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let n = labels.len() as f64;
    let mut label_counts = std::collections::HashMap::new();

    for &label in labels.iter() {
        *label_counts.entry(label).or_insert(0) += 1;
    }

    let mut h = F::zero();
    for &count in label_counts.values() {
        if count > 0 {
            let p = count as f64 / n;
            h = h - F::from(p * p.ln()).unwrap();
        }
    }

    Ok(h)
}

/// Build contingency matrix for two label assignments.
fn build_contingency_matrix(
    labels_true: ArrayView1<i32>,
    labels_pred: ArrayView1<i32>,
) -> Result<Array2<usize>> {
    let mut true_labels = std::collections::BTreeSet::new();
    let mut pred_labels = std::collections::BTreeSet::new();

    for &label in labels_true.iter() {
        true_labels.insert(label);
    }
    for &label in labels_pred.iter() {
        pred_labels.insert(label);
    }

    let true_label_map: std::collections::HashMap<_, _> = true_labels
        .iter()
        .enumerate()
        .map(|(i, &label)| (label, i))
        .collect();
    let pred_label_map: std::collections::HashMap<_, _> = pred_labels
        .iter()
        .enumerate()
        .map(|(i, &label)| (label, i))
        .collect();

    let mut contingency = Array2::<usize>::zeros((true_labels.len(), pred_labels.len()));
    for i in 0..labels_true.len() {
        let true_idx = true_label_map[&labels_true[i]];
        let pred_idx = pred_label_map[&labels_pred[i]];
        contingency[[true_idx, pred_idx]] += 1;
    }

    Ok(contingency)
}

/// Homogeneity, completeness and V-measure metrics for clustering evaluation.
///
/// These metrics are useful to evaluate the quality of clustering when ground truth is available.
/// - Homogeneity: each cluster contains only members of a single class.
/// - Completeness: all members of a given class are assigned to the same cluster.
/// - V-measure: harmonic mean of homogeneity and completeness.
///
/// All scores are between 0 and 1, where 1 indicates perfect clustering.
///
/// # Arguments
///
/// * `labels_true` - Ground truth cluster labels
/// * `labels_pred` - Predicted cluster labels
///
/// # Returns
///
/// Tuple of (homogeneity, completeness, v_measure)
///
/// # Example
///
/// ```
/// use ndarray::Array1;
/// use scirs2_cluster::metrics::homogeneity_completeness_v_measure;
///
/// let labels_true = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
/// let labels_pred = Array1::from_vec(vec![0, 0, 1, 1, 1, 1]);
///
/// let (h, c, v): (f64, f64, f64) = homogeneity_completeness_v_measure(labels_true.view(), labels_pred.view()).unwrap();
/// assert!(h > 0.5);  // Good homogeneity
/// assert!(c > 0.9);  // High completeness (all members of each class in single clusters)
/// ```
pub fn homogeneity_completeness_v_measure<F>(
    labels_true: ArrayView1<i32>,
    labels_pred: ArrayView1<i32>,
) -> Result<(F, F, F)>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    if labels_true.len() != labels_pred.len() {
        return Err(ClusteringError::InvalidInput(
            "Labels arrays must have the same length".to_string(),
        ));
    }

    let n = labels_true.len();
    if n == 0 {
        return Ok((F::one(), F::one(), F::one()));
    }

    // Compute entropies
    let h_true = entropy::<F>(labels_true)?;
    let h_pred = entropy::<F>(labels_pred)?;

    // Edge cases
    if h_true == F::zero() {
        return Ok((F::one(), F::one(), F::one()));
    }
    if h_pred == F::zero() {
        return Ok((F::one(), F::one(), F::one()));
    }

    // Compute conditional entropies
    let h_true_given_pred = conditional_entropy::<F>(labels_true, labels_pred)?;
    let h_pred_given_true = conditional_entropy::<F>(labels_pred, labels_true)?;

    // Compute homogeneity
    let homogeneity = if h_pred == F::zero() {
        F::one()
    } else {
        F::one() - h_true_given_pred / h_true
    };

    // Compute completeness
    let completeness = if h_true == F::zero() {
        F::one()
    } else {
        F::one() - h_pred_given_true / h_pred
    };

    // Compute V-measure
    let v_measure = if homogeneity + completeness == F::zero() {
        F::zero()
    } else {
        F::from(2.0).unwrap() * homogeneity * completeness / (homogeneity + completeness)
    };

    Ok((homogeneity, completeness, v_measure))
}

/// Compute conditional entropy H(X|Y).
fn conditional_entropy<F>(labels_x: ArrayView1<i32>, labels_y: ArrayView1<i32>) -> Result<F>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let n = labels_x.len() as f64;
    let contingency = build_contingency_matrix(labels_x, labels_y)?;

    let mut h_xy = F::zero();
    let col_sums = contingency.sum_axis(Axis(0));

    for j in 0..contingency.shape()[1] {
        let n_j = col_sums[j] as f64;
        if n_j > 0.0 {
            for i in 0..contingency.shape()[0] {
                let n_ij = contingency[[i, j]] as f64;
                if n_ij > 0.0 {
                    let term = n_ij / n * (n_ij / n_j).ln();
                    h_xy = h_xy - F::from(term).unwrap();
                }
            }
        }
    }

    Ok(h_xy)
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

    #[test]
    fn test_adjusted_rand_index() {
        // Perfect agreement
        let labels_true = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        let labels_pred = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        let ari: f64 = adjusted_rand_index(labels_true.view(), labels_pred.view()).unwrap();
        assert!((ari - 1.0).abs() < 1e-6);

        // Label permutation (still perfect agreement)
        let labels_pred2 = Array1::from_vec(vec![1, 1, 2, 2, 0, 0]);
        let ari2: f64 = adjusted_rand_index(labels_true.view(), labels_pred2.view()).unwrap();
        assert!((ari2 - 1.0).abs() < 1e-6);

        // Partial agreement
        let labels_pred3 = Array1::from_vec(vec![0, 0, 1, 1, 1, 1]);
        let ari3: f64 = adjusted_rand_index(labels_true.view(), labels_pred3.view()).unwrap();
        assert!(ari3 > 0.0 && ari3 < 1.0);
    }

    #[test]
    fn test_normalized_mutual_info() {
        // Perfect agreement
        let labels_true = Array1::from_vec(vec![0, 0, 1, 1]);
        let labels_pred = Array1::from_vec(vec![0, 0, 1, 1]);

        let nmi: f64 =
            normalized_mutual_info(labels_true.view(), labels_pred.view(), "arithmetic").unwrap();
        assert!((nmi - 1.0).abs() < 1e-6);

        // Test different normalizations
        let labels_true2 = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        let labels_pred2 = Array1::from_vec(vec![0, 0, 1, 1, 1, 1]);

        let nmi_arith: f64 =
            normalized_mutual_info(labels_true2.view(), labels_pred2.view(), "arithmetic").unwrap();
        let nmi_geom: f64 =
            normalized_mutual_info(labels_true2.view(), labels_pred2.view(), "geometric").unwrap();

        assert!(nmi_arith > 0.0 && nmi_arith < 1.0);
        assert!(nmi_geom > 0.0 && nmi_geom < 1.0);
    }

    #[test]
    fn test_homogeneity_completeness_v_measure() {
        // Perfect clustering
        let labels_true = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        let labels_pred = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);

        let (h, c, v): (f64, f64, f64) =
            homogeneity_completeness_v_measure(labels_true.view(), labels_pred.view()).unwrap();
        assert!((h - 1.0).abs() < 1e-6);
        assert!((c - 1.0).abs() < 1e-6);
        assert!((v - 1.0).abs() < 1e-6);

        // Imperfect clustering - classes 1 and 2 merged
        let labels_pred2 = Array1::from_vec(vec![0, 0, 1, 1, 1, 1]);
        let (h2, c2, v2): (f64, f64, f64) =
            homogeneity_completeness_v_measure(labels_true.view(), labels_pred2.view()).unwrap();

        // When classes are merged, completeness is actually perfect (1.0) because
        // all members of each true class are contained within single predicted clusters
        // Homogeneity is lower because predicted clusters contain multiple true classes
        assert!(h2 > 0.0 && h2 < 1.0); // Reduced homogeneity
        assert!(c2 > 0.9); // High completeness
        assert!(v2 > 0.0 && v2 < 1.0); // V-measure between 0 and 1
    }
}
