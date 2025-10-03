//! Contingency matrix utilities for clustering evaluation
//!
//! This module provides utilities for building and working with contingency matrices
//! used in clustering evaluation metrics.

use scirs2_core::ndarray::{Array2, ArrayView1};

/// Build a contingency matrix from two label arrays
///
/// The contingency matrix C has C[i, j] equal to the number of samples
/// that belong to class i in the true labels and class j in the predicted labels.
///
/// # Arguments
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Predicted cluster labels
///
/// # Returns
/// * `Result<Array2<usize>, String>` - The contingency matrix
pub fn build_contingency_matrix(
    labels_true: ArrayView1<i32>,
    labels_pred: ArrayView1<i32>,
) -> Result<Array2<usize>, String> {
    if labels_true.len() != labels_pred.len() {
        return Err("Labels arrays must have the same length".to_string());
    }

    // Find unique labels
    let mut unique_true = Vec::new();
    let mut unique_pred = Vec::new();

    for &label in labels_true.iter() {
        if !unique_true.contains(&label) {
            unique_true.push(label);
        }
    }

    for &label in labels_pred.iter() {
        if !unique_pred.contains(&label) {
            unique_pred.push(label);
        }
    }

    unique_true.sort();
    unique_pred.sort();

    let n_true = unique_true.len();
    let n_pred = unique_pred.len();

    let mut contingency = Array2::zeros((n_true, n_pred));

    for (&true_label, &pred_label) in labels_true.iter().zip(labels_pred.iter()) {
        if let (Some(i), Some(j)) = (
            unique_true.iter().position(|&x| x == true_label),
            unique_pred.iter().position(|&x| x == pred_label),
        ) {
            contingency[[i, j]] += 1;
        }
    }

    Ok(contingency)
}
