//! Rand-based metrics for clustering evaluation
//!
//! This module provides metrics based on the Rand index and its variants
//! for evaluating clustering quality. These metrics compare pairs of points
//! and measure agreement between true and predicted clusterings.

use ndarray::ArrayView1;
use num_traits::{Float, FromPrimitive};
use std::collections::HashMap;
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};

/// Calculate the Rand score between two label assignments
///
/// The Rand index measures the similarity between two clusterings by considering
/// all pairs of samples and counting pairs that are classified consistently
/// across both clusterings.
///
/// # Arguments
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Predicted cluster labels
///
/// # Returns
/// * `Result<F>` - The Rand score (0 to 1, higher is better)
///
/// # Example
/// ```rust
/// use ndarray::Array1;
/// use scirs2_cluster::metrics::information_theoretic::rand_score;
///
/// let true_labels = Array1::from_vec(vec![0, 0, 1, 1]);
/// let pred_labels = Array1::from_vec(vec![0, 1, 0, 1]);
///
/// let rand: f64 = rand_score(true_labels.view(), pred_labels.view()).unwrap();
/// assert!(rand >= 0.0 && rand <= 1.0);
/// ```
pub fn rand_score<F>(labels_true: ArrayView1<i32>, labels_pred: ArrayView1<i32>) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    if labels_true.len() != labels_pred.len() {
        return Err(ClusteringError::InvalidInput(
            "True and predicted labels must have the same length".to_string(),
        ));
    }

    let n = labels_true.len();
    if n <= 1 {
        return Ok(F::one());
    }

    let mut agreements = 0; // Pairs classified the same way in both clusterings

    for i in 0..n {
        for j in (i + 1)..n {
            let same_true = labels_true[i] == labels_true[j];
            let same_pred = labels_pred[i] == labels_pred[j];

            if same_true == same_pred {
                agreements += 1;
            }
        }
    }

    let total_pairs = n * (n - 1) / 2;
    if total_pairs == 0 {
        Ok(F::one())
    } else {
        Ok(F::from(agreements as f64 / total_pairs as f64).unwrap())
    }
}

/// Calculate the Adjusted Rand Index between two label assignments
///
/// The Adjusted Rand Index (ARI) is a corrected-for-chance version of the Rand index.
/// It has expected value zero for independent clusterings and maximum value 1 for
/// identical clusterings. It can be negative if the clustering is worse than random.
///
/// # Arguments
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Predicted cluster labels
///
/// # Returns
/// * `Result<F>` - The adjusted Rand index (-1 to 1, higher is better)
///
/// # Example
/// ```rust
/// use ndarray::Array1;
/// use scirs2_cluster::metrics::adjusted_rand_score;
///
/// let true_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
/// let pred_labels = Array1::from_vec(vec![0, 0, 1, 1, 1, 2]);
///
/// let ari: f64 = adjusted_rand_score(true_labels.view(), pred_labels.view()).unwrap();
/// assert!(ari >= -1.0 && ari <= 1.0);
/// ```
pub fn adjusted_rand_score<F>(
    labels_true: ArrayView1<i32>,
    labels_pred: ArrayView1<i32>,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    if labels_true.len() != labels_pred.len() {
        return Err(ClusteringError::InvalidInput(
            "True and predicted labels must have the same length".to_string(),
        ));
    }

    let n_samples = labels_true.len();
    if n_samples <= 1 {
        return Ok(F::one());
    }

    let contingency = build_contingency_table(labels_true, labels_pred);

    // Calculate marginal sums
    let mut row_sums = HashMap::new();
    let mut col_sums = HashMap::new();
    for (&(i, j), &count) in &contingency {
        *row_sums.entry(i).or_insert(0) += count;
        *col_sums.entry(j).or_insert(0) += count;
    }

    // Calculate ARI components
    let mut sum_comb_c = F::zero(); // Sum of combinations C(n_ij, 2)
    for &n_ij in contingency.values() {
        if n_ij >= 2 {
            sum_comb_c = sum_comb_c + F::from(combination(n_ij, 2)).unwrap();
        }
    }

    let mut sum_comb_a = F::zero(); // Sum of combinations C(n_i, 2) for rows
    for &n_i in row_sums.values() {
        if n_i >= 2 {
            sum_comb_a = sum_comb_a + F::from(combination(n_i, 2)).unwrap();
        }
    }

    let mut sum_comb_b = F::zero(); // Sum of combinations C(n_j, 2) for columns
    for &n_j in col_sums.values() {
        if n_j >= 2 {
            sum_comb_b = sum_comb_b + F::from(combination(n_j, 2)).unwrap();
        }
    }

    let comb_n = F::from(combination(n_samples, 2)).unwrap();

    // Calculate ARI using the standard formula
    let expected_index = (sum_comb_a * sum_comb_b) / comb_n;
    let max_index = (sum_comb_a + sum_comb_b) / F::from(2).unwrap();

    let denominator = max_index - expected_index;

    if denominator.abs() < F::from(1e-10).unwrap() {
        Ok(F::zero())
    } else {
        Ok((sum_comb_c - expected_index) / denominator)
    }
}

/// Calculate the Fowlkes-Mallows score between two label assignments
///
/// The Fowlkes-Mallows index is the geometric mean of precision and recall
/// for pairs of points. It measures the similarity between two clusterings
/// based on pair-wise decisions.
///
/// # Arguments
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Predicted cluster labels
///
/// # Returns
/// * `Result<F>` - The Fowlkes-Mallows score (0 to 1, higher is better)
pub fn fowlkes_mallows_score<F>(
    labels_true: ArrayView1<i32>,
    labels_pred: ArrayView1<i32>,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    if labels_true.len() != labels_pred.len() {
        return Err(ClusteringError::InvalidInput(
            "True and predicted labels must have the same length".to_string(),
        ));
    }

    let n = labels_true.len();
    if n <= 1 {
        return Ok(F::one());
    }

    let mut tp = 0; // True positives: pairs in same cluster in both
    let mut fp = 0; // False positives: pairs in same cluster in pred but not true
    let mut fn_count = 0; // False negatives: pairs in same cluster in true but not pred

    for i in 0..n {
        for j in (i + 1)..n {
            let same_true = labels_true[i] == labels_true[j];
            let same_pred = labels_pred[i] == labels_pred[j];

            match (same_true, same_pred) {
                (true, true) => tp += 1,
                (false, true) => fp += 1,
                (true, false) => fn_count += 1,
                (false, false) => {} // True negatives don't contribute to F-M index
            }
        }
    }

    if tp == 0 {
        return Ok(F::zero());
    }

    let precision = tp as f64 / (tp + fp) as f64;
    let recall = tp as f64 / (tp + fn_count) as f64;

    if precision == 0.0 || recall == 0.0 {
        Ok(F::zero())
    } else {
        Ok(F::from((precision * recall).sqrt()).unwrap())
    }
}

/// Compute pair confusion matrix for clustering evaluation
///
/// This function computes the confusion matrix for pairs of points,
/// which is useful for computing various pair-based clustering metrics.
///
/// # Arguments
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Predicted cluster labels
///
/// # Returns
/// * `Result<(usize, usize, usize, usize)>` - Tuple of (TP, TN, FP, FN) counts
pub fn pair_confusion_matrix(
    labels_true: ArrayView1<i32>,
    labels_pred: ArrayView1<i32>,
) -> Result<(usize, usize, usize, usize)> {
    if labels_true.len() != labels_pred.len() {
        return Err(ClusteringError::InvalidInput(
            "True and predicted labels must have the same length".to_string(),
        ));
    }

    let n = labels_true.len();
    let mut tp = 0; // True positives
    let mut tn = 0; // True negatives
    let mut fp = 0; // False positives
    let mut fn_count = 0; // False negatives

    for i in 0..n {
        for j in (i + 1)..n {
            let same_true = labels_true[i] == labels_true[j];
            let same_pred = labels_pred[i] == labels_pred[j];

            match (same_true, same_pred) {
                (true, true) => tp += 1,
                (false, false) => tn += 1,
                (false, true) => fp += 1,
                (true, false) => fn_count += 1,
            }
        }
    }

    Ok((tp, tn, fp, fn_count))
}

/// Compute precision and recall for pair-wise clustering decisions
///
/// # Arguments
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Predicted cluster labels
///
/// # Returns
/// * `Result<(F, F)>` - Tuple of (precision, recall)
pub fn pair_precision_recall<F>(
    labels_true: ArrayView1<i32>,
    labels_pred: ArrayView1<i32>,
) -> Result<(F, F)>
where
    F: Float + FromPrimitive + Debug,
{
    let (tp, _tn, fp, fn_count) = pair_confusion_matrix(labels_true, labels_pred)?;

    let precision = if tp + fp > 0 {
        F::from(tp as f64 / (tp + fp) as f64).unwrap()
    } else {
        F::zero()
    };

    let recall = if tp + fn_count > 0 {
        F::from(tp as f64 / (tp + fn_count) as f64).unwrap()
    } else {
        F::zero()
    };

    Ok((precision, recall))
}

/// Build contingency table from two label arrays
fn build_contingency_table(
    labels_true: ArrayView1<i32>,
    labels_pred: ArrayView1<i32>,
) -> HashMap<(i32, i32), usize> {
    let mut contingency = HashMap::new();
    for (&true_label, &pred_label) in labels_true.iter().zip(labels_pred.iter()) {
        *contingency.entry((true_label, pred_label)).or_insert(0) += 1;
    }
    contingency
}

/// Calculate binomial coefficient C(n, k)
fn combination(n: usize, k: usize) -> usize {
    if k > n || k == 0 {
        return if k == 0 { 1 } else { 0 };
    }

    if k == 1 {
        return n;
    }

    if k == 2 {
        return n * (n - 1) / 2;
    }

    // For larger k, use the standard formula
    let k = k.min(n - k); // Take advantage of symmetry
    let mut result = 1;

    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_rand_score_identical() {
        let labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        let rand: f64 = rand_score(labels.view(), labels.view()).unwrap();
        assert!((rand - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_adjusted_rand_score_identical() {
        let labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        let ari: f64 = adjusted_rand_score(labels.view(), labels.view()).unwrap();
        assert!((ari - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_adjusted_rand_score_random() {
        let true_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        let pred_labels = Array1::from_vec(vec![0, 1, 2, 0, 1, 2]);

        let ari: f64 = adjusted_rand_score(true_labels.view(), pred_labels.view()).unwrap();
        assert!(ari >= -1.0 && ari <= 1.0);
    }

    #[test]
    fn test_fowlkes_mallows_score() {
        let true_labels = Array1::from_vec(vec![0, 0, 1, 1]);
        let pred_labels = Array1::from_vec(vec![0, 1, 0, 1]);

        let fm: f64 = fowlkes_mallows_score(true_labels.view(), pred_labels.view()).unwrap();
        assert!(fm >= 0.0 && fm <= 1.0);
    }

    #[test]
    fn test_pair_confusion_matrix() {
        let true_labels = Array1::from_vec(vec![0, 0, 1, 1]);
        let pred_labels = Array1::from_vec(vec![0, 1, 0, 1]);

        let (tp, tn, fp, fn_count) =
            pair_confusion_matrix(true_labels.view(), pred_labels.view()).unwrap();

        // Total pairs should be C(4,2) = 6
        assert_eq!(tp + tn + fp + fn_count, 6);
    }

    #[test]
    fn test_pair_precision_recall() {
        let true_labels = Array1::from_vec(vec![0, 0, 1, 1]);
        let pred_labels = Array1::from_vec(vec![0, 0, 1, 1]);

        let (precision, recall): (f64, f64) =
            pair_precision_recall(true_labels.view(), pred_labels.view()).unwrap();

        // Perfect clustering should have precision and recall of 1.0
        assert!((precision - 1.0).abs() < 1e-10);
        assert!((recall - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_combination() {
        assert_eq!(combination(4, 2), 6);
        assert_eq!(combination(5, 3), 10);
        assert_eq!(combination(6, 0), 1);
        assert_eq!(combination(6, 6), 1);
        assert_eq!(combination(3, 5), 0); // k > n
    }
}
