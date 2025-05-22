//! Homogeneity, Completeness, and V-measure metrics
//!
//! This module provides metrics for evaluating clustering quality based on class assignment.
//! Homogeneity measures how each cluster contains only members of a single class.
//! Completeness measures how all members of a given class are assigned to the same cluster.
//! V-measure is the harmonic mean of homogeneity and completeness.

use ndarray::{ArrayBase, Data, Dimension};
use std::collections::HashMap;
use std::fmt::Debug;

use crate::error::{MetricsError, Result};

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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_homogeneity_completeness_v_measure() {
        // Perfect match
        let labels_true = array![0, 0, 1, 1, 2, 2];
        let labels_pred = array![1, 1, 0, 0, 2, 2]; // Same clustering with different labels

        let (h, c, v) =
            homogeneity_completeness_v_measure(&labels_true, &labels_pred, 1.0).unwrap();

        assert!((h - 1.0).abs() < 1e-10);
        assert!((c - 1.0).abs() < 1e-10);
        assert!((v - 1.0).abs() < 1e-10);

        // Partial match
        let labels_true = array![0, 0, 1, 1, 2, 2];
        let labels_pred = array![0, 0, 0, 1, 1, 1];

        let (h, c, v) =
            homogeneity_completeness_v_measure(&labels_true, &labels_pred, 1.0).unwrap();

        assert!(h > 0.0 && h < 1.0);
        assert!(c > 0.0 && c < 1.0);
        assert!(v > 0.0 && v < 1.0);
    }
}
