//! Information-theoretic clustering evaluation metrics
//!
//! This module provides metrics based on information theory for evaluating
//! clustering quality, including Jensen-Shannon divergence, variation of information,
//! and mutual information-based measures.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use num_traits::{Float, FromPrimitive};
use std::collections::HashMap;
use std::fmt::Debug;
use std::ops::{AddAssign, DivAssign, SubAssign};

use crate::error::{ClusteringError, Result};
use crate::utils::contingency::build_contingency_matrix;

/// Jensen-Shannon divergence between two clusterings.
///
/// The Jensen-Shannon divergence is a symmetric and bounded measure
/// of similarity between two probability distributions. It's based on
/// the Kullback-Leibler divergence but is symmetric and always finite.
///
/// # Arguments
/// * `labels_true` - Ground truth cluster labels
/// * `labels_pred` - Predicted cluster labels
///
/// # Returns
/// * `Result<F>` - The Jensen-Shannon distance (square root of divergence)
///
/// # Example
/// ```rust
/// use ndarray::Array1;
/// use scirs2_cluster::metrics::jensen_shannon_divergence;
///
/// let labels_true = Array1::from_vec(vec![0, 0, 1, 1]);
/// let labels_pred = Array1::from_vec(vec![0, 1, 0, 1]);
/// let js: f64 = jensen_shannon_divergence(labels_true.view(), labels_pred.view()).unwrap();
/// ```
pub fn jensen_shannon_divergence<F>(
    labels_true: ArrayView1<i32>,
    labels_pred: ArrayView1<i32>,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug + AddAssign + 'static,
{
    if labels_true.len() != labels_pred.len() {
        return Err(ClusteringError::InvalidInput(
            "Labels arrays must have the same length".to_string(),
        ));
    }

    let n = labels_true.len();
    if n == 0 {
        return Ok(F::zero());
    }

    // Convert labels to probability distributions
    let p = label_distribution::<F>(labels_true)?;
    let q = label_distribution::<F>(labels_pred)?;

    // Compute average distribution
    let mut m = HashMap::new();
    for (label, &prob) in &p {
        *m.entry(*label).or_insert(F::zero()) += prob / F::from(2.0).unwrap();
    }
    for (label, &prob) in &q {
        *m.entry(*label).or_insert(F::zero()) += prob / F::from(2.0).unwrap();
    }

    // Compute KL divergences
    let kl_pm = kl_divergence(&p, &m)?;
    let kl_qm = kl_divergence(&q, &m)?;

    // Jensen-Shannon divergence
    let js = (kl_pm + kl_qm) / F::from(2.0).unwrap();
    Ok(js.sqrt()) // Return the Jensen-Shannon distance
}

/// Variation of Information (VI) between two clusterings.
///
/// The Variation of Information is a symmetric measure that equals the sum of
/// conditional entropies H(X|Y) + H(Y|X).
///
/// # Arguments
/// * `labels_true` - Ground truth cluster labels
/// * `labels_pred` - Predicted cluster labels
///
/// # Returns
/// * `Result<F>` - The Variation of Information score (lower is better)
///
/// # Example
/// ```rust
/// use ndarray::Array1;
/// use scirs2_cluster::metrics::variation_of_information;
///
/// let labels_true = Array1::from_vec(vec![0, 0, 1, 1]);
/// let labels_pred = Array1::from_vec(vec![0, 1, 0, 1]);
/// let vi: f64 = variation_of_information(labels_true.view(), labels_pred.view()).unwrap();
/// ```
pub fn variation_of_information<F>(
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

    let h_true_given_pred = conditional_entropy::<F>(labels_true, labels_pred)?;
    let h_pred_given_true = conditional_entropy::<F>(labels_pred, labels_true)?;

    Ok(h_true_given_pred + h_pred_given_true)
}

/// Information-theoretic cluster quality measure.
///
/// This measure combines intra-cluster and inter-cluster information
/// to evaluate clustering quality without ground truth.
///
/// # Arguments
/// * `data` - Input data (n_samples x n_features)
/// * `labels` - Cluster labels
///
/// # Returns
/// * `Result<F>` - The information-theoretic quality score (higher is better)
pub fn information_cluster_quality<F>(data: ArrayView2<F>, labels: ArrayView1<i32>) -> Result<F>
where
    F: Float + FromPrimitive + Debug + PartialOrd + AddAssign + SubAssign + DivAssign + 'static,
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
        return Ok(F::zero());
    }

    // Compute cluster entropies (within-cluster information)
    let mut total_within_cluster_entropy = F::zero();
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

        if cluster_indices.len() < 2 {
            continue;
        }

        // Calculate feature-wise entropy within cluster
        let mut cluster_entropy = F::zero();
        for feature_idx in 0..n_features {
            let mut feature_values = Vec::new();
            for &idx in &cluster_indices {
                feature_values.push(data[[idx, feature_idx]]);
            }

            // Discretize continuous values into bins for entropy calculation
            let entropy = calculate_entropy(&feature_values)?;
            cluster_entropy = cluster_entropy + entropy;
        }

        total_within_cluster_entropy = total_within_cluster_entropy
            + cluster_entropy * F::from_usize(cluster_indices.len()).unwrap();
    }

    // Normalize by total samples
    total_within_cluster_entropy = total_within_cluster_entropy / F::from_usize(n_samples).unwrap();

    // Higher entropy within clusters indicates worse clustering
    // Return inverse of entropy as quality measure
    if total_within_cluster_entropy > F::zero() {
        Ok(F::one() / total_within_cluster_entropy)
    } else {
        Ok(F::infinity())
    }
}

/// Calculate entropy of a set of values by discretizing them into bins.
fn calculate_entropy<F>(values: &[F]) -> Result<F>
where
    F: Float + FromPrimitive + Debug + PartialOrd + 'static,
{
    if values.is_empty() {
        return Ok(F::zero());
    }

    if values.len() == 1 {
        return Ok(F::zero());
    }

    // Find min and max values
    let mut min_val = values[0];
    let mut max_val = values[0];
    for &val in values.iter() {
        if val < min_val {
            min_val = val;
        }
        if val > max_val {
            max_val = val;
        }
    }

    // Use sqrt(n) bins as a heuristic
    let n_bins = ((values.len() as f64).sqrt() as usize).max(2);
    let range = max_val - min_val;

    if range == F::zero() {
        return Ok(F::zero());
    }

    let bin_width = range / F::from_usize(n_bins).unwrap();

    // Count values in each bin
    let mut bin_counts = vec![0; n_bins];
    for &val in values.iter() {
        let bin_idx = if val == max_val {
            n_bins - 1
        } else {
            let normalized = (val - min_val) / bin_width;
            normalized.to_usize().unwrap_or(0).min(n_bins - 1)
        };
        bin_counts[bin_idx] += 1;
    }

    // Calculate entropy
    let n_total = F::from_usize(values.len()).unwrap();
    let mut entropy = F::zero();
    for &count in &bin_counts {
        if count > 0 {
            let prob = F::from_usize(count).unwrap() / n_total;
            entropy = entropy - prob * prob.ln();
        }
    }

    Ok(entropy)
}

/// Convert cluster labels to a probability distribution.
fn label_distribution<F>(labels: ArrayView1<i32>) -> Result<HashMap<i32, F>>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let mut counts = HashMap::new();
    let mut total = 0;

    for &label in labels.iter() {
        if label >= 0 {
            *counts.entry(label).or_insert(0) += 1;
            total += 1;
        }
    }

    if total == 0 {
        return Ok(HashMap::new());
    }

    let mut distribution = HashMap::new();
    for (label, count) in counts {
        distribution.insert(label, F::from(count).unwrap() / F::from(total).unwrap());
    }

    Ok(distribution)
}

/// Calculate Kullback-Leibler divergence between two probability distributions.
fn kl_divergence<F>(p: &HashMap<i32, F>, q: &HashMap<i32, F>) -> Result<F>
where
    F: Float + FromPrimitive + Debug + AddAssign + 'static,
{
    let mut kl = F::zero();
    for (label, &p_val) in p {
        if p_val > F::zero() {
            let q_val = q.get(label).cloned().unwrap_or(F::from(1e-10).unwrap()); // Smoothing
            if q_val > F::zero() {
                kl += p_val * (p_val / q_val).ln();
            }
        }
    }
    Ok(kl)
}

/// Calculate conditional entropy H(X|Y).
fn conditional_entropy<F>(labels_x: ArrayView1<i32>, labels_y: ArrayView1<i32>) -> Result<F>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let n = labels_x.len() as f64;
    let contingency = build_contingency_matrix(labels_x, labels_y)
        .map_err(|e| ClusteringError::InvalidInput(e))?;
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
    use ndarray::Array2;

    #[test]
    fn test_jensen_shannon_divergence() {
        let labels_true = Array1::from_vec(vec![0, 0, 1, 1]);
        let labels_pred = Array1::from_vec(vec![0, 1, 0, 1]);

        let js: f64 = jensen_shannon_divergence(labels_true.view(), labels_pred.view()).unwrap();
        assert!(js >= 0.0 && js <= 1.0);
    }

    #[test]
    fn test_variation_of_information() {
        let labels_true = Array1::from_vec(vec![0, 0, 1, 1]);
        let labels_pred = Array1::from_vec(vec![0, 1, 0, 1]);

        let vi: f64 = variation_of_information(labels_true.view(), labels_pred.view()).unwrap();
        assert!(vi >= 0.0);
    }

    #[test]
    fn test_information_cluster_quality() {
        let data = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 1.0, 10.0, 10.0, 11.0, 11.0])
            .unwrap();
        let labels = Array1::from_vec(vec![0, 0, 1, 1]);

        let quality = information_cluster_quality(data.view(), labels.view()).unwrap();
        assert!(quality.is_finite() && quality > 0.0);
    }

    #[test]
    fn test_calculate_entropy() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let entropy = calculate_entropy(&values).unwrap();
        assert!(entropy > 0.0);
    }

    #[test]
    fn test_label_distribution() {
        let labels = Array1::from_vec(vec![0, 0, 1, 1, 1]);
        let dist = label_distribution::<f64>(labels.view()).unwrap();

        assert!((dist[&0] - 0.4).abs() < 1e-10);
        assert!((dist[&1] - 0.6).abs() < 1e-10);
    }
}
