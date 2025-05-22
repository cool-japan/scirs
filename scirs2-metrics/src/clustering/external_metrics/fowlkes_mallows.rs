//! Fowlkes-Mallows score for cluster evaluation
//!
//! This module provides the Fowlkes-Mallows score for evaluating clustering quality
//! by comparing the predicted clusters with ground truth.

use ndarray::{ArrayBase, Data, Dimension};
use std::collections::HashMap;
use std::fmt::Debug;

use crate::error::{MetricsError, Result};

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
    T: Clone + std::hash::Hash + Eq + Debug,
    U: Clone + std::hash::Hash + Eq + Debug,
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

/// Helper function to calculate nC2 (number of ways to choose 2 items from n)
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
    use ndarray::array;

    #[test]
    fn test_fowlkes_mallows_score() {
        // Perfect match
        let labels_true = array![0, 0, 1, 1, 2, 2];
        let labels_pred = array![1, 1, 0, 0, 2, 2]; // Same clustering with different labels

        let score = fowlkes_mallows_score(&labels_true, &labels_pred).unwrap();
        assert!((score - 1.0).abs() < 1e-10);

        // Partial match
        let labels_true = array![0, 0, 1, 1, 2, 2];
        let labels_pred = array![0, 0, 0, 1, 1, 1];

        let score = fowlkes_mallows_score(&labels_true, &labels_pred).unwrap();
        assert!(score > 0.0 && score < 1.0);
    }
}
