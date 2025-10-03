//! One-vs-One Classification Metrics
//!
//! This module provides implementation of one-vs-one (OvO) classification metrics.
//! In one-vs-one classification, a binary classifier is trained for each pair of classes,
//! resulting in n*(n-1)/2 binary classifiers for n classes. The metrics in this module
//! help evaluate the performance of such multi-class classification strategies.
//!
//! ## Main Metrics
//!
//! - `one_vs_one_accuracy`: Computes the average accuracy across all possible pairwise class combinations
//! - `one_vs_one_precision_recall`: Computes precision and recall for each class in OvO setting
//! - `one_vs_one_f1_score`: Computes F1 score for each class in OvO setting

use scirs2_core::ndarray::{ArrayBase, Data, Dimension};
use scirs2_core::numeric::NumCast;
use std::collections::{BTreeSet, HashMap};

use crate::error::{MetricsError, Result};

/// Calculates one-vs-one accuracy score for multi-class classification
///
/// In one-vs-one strategy, a binary classifier is trained for each pair of classes.
/// This function calculates the average accuracy across all pairwise classifiers.
///
/// # Arguments
///
/// * `y_true` - Ground truth labels
/// * `y_pred` - Predicted labels
///
/// # Returns
///
/// * Average accuracy across all pairwise classifiers (float between 0 and 1)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_metrics::classification::one_vs_one::one_vs_one_accuracy;
///
/// let y_true = array![0, 1, 2, 0, 1, 2];
/// let y_pred = array![0, 2, 1, 0, 0, 2];
///
/// let ovo_acc = one_vs_one_accuracy(&y_true, &y_pred).unwrap();
/// ```
#[allow(dead_code)]
pub fn one_vs_one_accuracy<T, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
) -> Result<f64>
where
    T: PartialEq + NumCast + Clone + Ord,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    D1: Dimension,
    D2: Dimension,
{
    // Check that arrays have the same shape
    if y_true.shape() != y_pred.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_pred have different shapes: {:?} vs {:?}",
            y_true.shape(),
            y_pred.shape()
        )));
    }

    let n_samples = y_true.len();
    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    // Get unique classes from _true labels
    let mut classes: BTreeSet<T> = BTreeSet::new();
    for label in y_true.iter() {
        classes.insert(label.clone());
    }
    let classes_vec: Vec<T> = classes.into_iter().collect();
    let n_classes = classes_vec.len();

    // If there's only one class, return 1.0
    if n_classes <= 1 {
        return Ok(1.0);
    }

    let mut total_accuracy = 0.0;
    let mut pair_count = 0;

    // Compute accuracy for each pair of classes
    for i in 0..n_classes {
        for j in (i + 1)..n_classes {
            let class_i = &classes_vec[i];
            let class_j = &classes_vec[j];

            // Create mask for samples that are either class i or class j
            let mut relevant_indices = Vec::new();
            for (idx, label) in y_true.iter().enumerate() {
                if label == class_i || label == class_j {
                    relevant_indices.push(idx);
                }
            }

            // Skip if no samples for this pair
            if relevant_indices.is_empty() {
                continue;
            }

            // Count correct classifications for this pair
            let mut correct = 0;
            for &idx in &relevant_indices {
                if y_true.iter().nth(idx) == y_pred.iter().nth(idx) {
                    correct += 1;
                }
            }

            // Calculate accuracy for this pair
            let pair_accuracy = correct as f64 / relevant_indices.len() as f64;
            total_accuracy += pair_accuracy;
            pair_count += 1;
        }
    }

    // Handle the case where there are no valid pairs
    if pair_count == 0 {
        return Ok(0.0);
    }

    // Return the average accuracy across all pairs
    Ok(total_accuracy / pair_count as f64)
}

/// Calculates one-vs-one precision and recall scores for each class
///
/// In one-vs-one strategy, this calculates precision and recall for each class
/// by considering each binary problem separately and then averaging.
///
/// # Arguments
///
/// * `y_true` - Ground truth labels
/// * `y_pred` - Predicted labels
///
/// # Returns
///
/// * Tuple of (precision_per_class, recall_per_class) where each is a HashMap from class to score
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_metrics::classification::one_vs_one::one_vs_one_precision_recall;
///
/// let y_true = array![0, 1, 2, 0, 1, 2];
/// let y_pred = array![0, 2, 1, 0, 0, 2];
///
/// let (precision, recall) = one_vs_one_precision_recall(&y_true, &y_pred).unwrap();
/// ```
#[allow(dead_code)]
pub fn one_vs_one_precision_recall<T, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
) -> Result<(HashMap<T, f64>, HashMap<T, f64>)>
where
    T: PartialEq + NumCast + Clone + Ord + std::hash::Hash,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    D1: Dimension,
    D2: Dimension,
{
    // Check that arrays have the same shape
    if y_true.shape() != y_pred.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_pred have different shapes: {:?} vs {:?}",
            y_true.shape(),
            y_pred.shape()
        )));
    }

    let n_samples = y_true.len();
    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    // Get unique classes
    let mut classes: BTreeSet<T> = BTreeSet::new();
    for label in y_true.iter() {
        classes.insert(label.clone());
    }
    for label in y_pred.iter() {
        classes.insert(label.clone());
    }
    let classes_vec: Vec<T> = classes.into_iter().collect();
    let n_classes = classes_vec.len();

    // Initialize result maps
    let mut precision_per_class: HashMap<T, f64> = HashMap::new();
    let mut recall_per_class: HashMap<T, f64> = HashMap::new();

    // Compute precision and recall for each class
    for i in 0..n_classes {
        let current_class = &classes_vec[i];

        // Initialize counters for this class
        let mut true_positives = 0;
        let mut false_positives = 0;
        let mut false_negatives = 0;

        // Count TP, FP, and FN
        for (true_label, pred_label) in y_true.iter().zip(y_pred.iter()) {
            if true_label == current_class && pred_label == current_class {
                true_positives += 1;
            } else if true_label != current_class && pred_label == current_class {
                false_positives += 1;
            } else if true_label == current_class && pred_label != current_class {
                false_negatives += 1;
            }
        }

        // Calculate precision: TP / (TP + FP)
        let precision = if true_positives + false_positives > 0 {
            true_positives as f64 / (true_positives + false_positives) as f64
        } else {
            0.0
        };

        // Calculate recall: TP / (TP + FN)
        let recall = if true_positives + false_negatives > 0 {
            true_positives as f64 / (true_positives + false_negatives) as f64
        } else {
            0.0
        };

        // Store in result maps
        precision_per_class.insert(current_class.clone(), precision);
        recall_per_class.insert(current_class.clone(), recall);
    }

    Ok((precision_per_class, recall_per_class))
}

/// Calculates one-vs-one F1 score for each class
///
/// F1 score is the harmonic mean of precision and recall.
/// This function computes F1 score for each class using the one-vs-one approach.
///
/// # Arguments
///
/// * `y_true` - Ground truth labels
/// * `y_pred` - Predicted labels
///
/// # Returns
///
/// * HashMap mapping each class to its F1 score
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_metrics::classification::one_vs_one::one_vs_one_f1_score;
///
/// let y_true = array![0, 1, 2, 0, 1, 2];
/// let y_pred = array![0, 2, 1, 0, 0, 2];
///
/// let f1_scores = one_vs_one_f1_score(&y_true, &y_pred).unwrap();
/// ```
#[allow(dead_code)]
pub fn one_vs_one_f1_score<T, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
) -> Result<HashMap<T, f64>>
where
    T: PartialEq + NumCast + Clone + Ord + std::hash::Hash,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    D1: Dimension,
    D2: Dimension,
{
    // Get precision and recall for each class
    let (precision_per_class, recall_per_class) = one_vs_one_precision_recall(y_true, y_pred)?;

    // Calculate F1 score for each class
    let mut f1_per_class: HashMap<T, f64> = HashMap::new();

    for (class, precision) in precision_per_class.iter() {
        let recall = recall_per_class.get(class).unwrap_or(&0.0);

        // F1 = 2 * (precision * recall) / (precision + recall)
        let f1 = if *precision + *recall > 0.0 {
            2.0 * (*precision * *recall) / (*precision + *recall)
        } else {
            0.0
        };

        f1_per_class.insert(class.clone(), f1);
    }

    Ok(f1_per_class)
}

/// Calculates weighted one-vs-one F1 score
///
/// Computes F1 score for each class and returns the weighted average
/// based on the number of true instances for each class.
///
/// # Arguments
///
/// * `y_true` - Ground truth labels
/// * `y_pred` - Predicted labels
///
/// # Returns
///
/// * Weighted average of F1 scores across all classes
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_metrics::classification::one_vs_one::weighted_one_vs_one_f1_score;
///
/// let y_true = array![0, 1, 2, 0, 1, 2];
/// let y_pred = array![0, 2, 1, 0, 0, 2];
///
/// let weighted_f1 = weighted_one_vs_one_f1_score(&y_true, &y_pred).unwrap();
/// ```
#[allow(dead_code)]
pub fn weighted_one_vs_one_f1_score<T, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
) -> Result<f64>
where
    T: PartialEq + NumCast + Clone + Ord + std::hash::Hash,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    D1: Dimension,
    D2: Dimension,
{
    // Get F1 score for each class
    let f1_per_class = one_vs_one_f1_score(y_true, y_pred)?;

    // Count instances of each class in _true labels
    let mut class_counts: HashMap<T, usize> = HashMap::new();
    for label in y_true.iter() {
        *class_counts.entry(label.clone()).or_insert(0) += 1;
    }

    // Calculate weighted average
    let mut weighted_sum = 0.0;
    let mut total_weight = 0;

    for (class, f1) in f1_per_class.iter() {
        let count = class_counts.get(class).unwrap_or(&0);
        weighted_sum += *f1 * (*count as f64);
        total_weight += *count;
    }

    if total_weight == 0 {
        return Ok(0.0);
    }

    Ok(weighted_sum / total_weight as f64)
}
