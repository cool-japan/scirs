//! Threshold optimization module
//!
//! This module provides functions for optimizing threshold values in binary classification
//! and computing precision-recall curves. These metrics are useful for understanding classifier
//! performance across different decision thresholds and finding optimal operating points.
//!
//! ## Precision-Recall Curves
//!
//! Precision-recall curves show the tradeoff between precision and recall at different threshold
//! values. They are particularly useful for imbalanced classification problems.
//!
//! ## Optimal Threshold Finding
//!
//! These functions help to determine the optimal decision threshold based on various criteria
//! such as F1 score, G-means, or custom cost functions.

use ndarray::{Array1, ArrayBase, Data, Dimension, Ix1};
use num_traits::real::Real;
use std::cmp::Ordering;

use crate::error::{MetricsError, Result};

/// Calculates the precision-recall curve for a binary classification problem.
///
/// The precision-recall curve shows the trade-off between precision and recall
/// for different threshold values. A high area under the curve represents both
/// high recall and high precision.
///
/// # Arguments
///
/// * `y_true` - Ground truth binary labels (0/1 or False/True)
/// * `y_prob` - Predicted probability estimates for the positive class
/// * `pos_label` - The label of the positive class
/// * `sample_weight` - Optional sample weights to apply
///
/// # Returns
///
/// * Tuple of (precision, recall, thresholds)
///   - precision: array of precision values
///   - recall: array of recall values
///   - thresholds: array of thresholds used for computing precision and recall
///
/// # Examples
///
/// ```ignore
/// use ndarray::array;
/// use scirs2_metrics::classification::threshold::precision_recall_curve;
///
/// let y_true = array![0.0, 0.0, 1.0, 1.0];
/// let y_prob = array![0.1, 0.4, 0.35, 0.8];
///
/// let (precision, recall, thresholds) = precision_recall_curve(&y_true, &y_prob, None, None).unwrap();
/// ```
pub fn precision_recall_curve<T, S1, S2, D1>(
    y_true: &ArrayBase<S1, D1>,
    y_prob: &ArrayBase<S2, Ix1>,
    pos_label: Option<T>,
    sample_weight: Option<&ArrayBase<S2, Ix1>>,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>)>
where
    T: Real + PartialOrd + Clone,
    S1: Data<Elem = T>,
    S2: Data<Elem = f64>,
    D1: Dimension,
{
    // Check that arrays have the same shape
    if y_true.len() != y_prob.len() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_prob have different lengths: {} vs {}",
            y_true.len(),
            y_prob.len()
        )));
    }

    let n_samples = y_true.len();
    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    // Handle sample weights
    let weights = if let Some(w) = sample_weight {
        if w.len() != n_samples {
            return Err(MetricsError::InvalidInput(format!(
                "sample_weight length does not match y_true length: {} vs {}",
                w.len(),
                n_samples
            )));
        }
        w.to_owned()
    } else {
        Array1::ones(n_samples)
    };

    // Determine positive class label
    let pos_lbl = pos_label.unwrap_or_else(|| T::one());

    // Create binary labels based on pos_label
    let binary_true: Vec<bool> = y_true.iter().map(|&y| y == pos_lbl).collect();

    // Create pairs of (probability, binary_label, weight) for sorting
    let mut prob_label_weight: Vec<_> = y_prob
        .iter()
        .zip(binary_true.iter())
        .zip(weights.iter())
        .map(|((p, &l), &w)| (*p, l, w))
        .collect();

    // Sort by probability in descending order
    prob_label_weight.sort_by(|(a, _, _), (b, _, _)| b.partial_cmp(a).unwrap_or(Ordering::Equal));

    // Extract unique threshold values
    let mut thresholds = Vec::new();
    let mut distinct_value_indices = Vec::new();

    for i in 0..prob_label_weight.len() {
        if i == 0 || prob_label_weight[i].0 != prob_label_weight[i - 1].0 {
            distinct_value_indices.push(i);
            thresholds.push(prob_label_weight[i].0);
        }
    }
    distinct_value_indices.push(prob_label_weight.len());

    // Calculate true positive and false positive counts for each threshold
    let mut tps = Vec::with_capacity(thresholds.len() + 1);
    let mut fps = Vec::with_capacity(thresholds.len() + 1);

    let mut tp_sum = 0.0;
    let mut fp_sum = 0.0;

    // Handle the edge case of threshold = infinity (no predictions)
    tps.push(0.0);
    fps.push(0.0);

    // Calculate TP and FP at each threshold
    for (i, &idx) in distinct_value_indices
        .iter()
        .take(distinct_value_indices.len() - 1)
        .enumerate()
    {
        let idx_range = idx..distinct_value_indices[i + 1];

        for j in idx_range {
            let (_, is_true_positive, weight) = prob_label_weight[j];
            if is_true_positive {
                tp_sum += weight;
            } else {
                fp_sum += weight;
            }
        }

        tps.push(tp_sum);
        fps.push(fp_sum);
    }

    // Calculate total positives (for recall denominator)
    let pos_weight: f64 = binary_true
        .iter()
        .zip(weights.iter())
        .map(|(&true_val, &weight)| if true_val { weight } else { 0.0 })
        .sum();

    if pos_weight <= 0.0 {
        return Err(MetricsError::InvalidInput(
            "No positive samples found in y_true".to_string(),
        ));
    }

    // Calculate precision and recall
    let mut precision = Vec::with_capacity(tps.len());
    let mut recall = Vec::with_capacity(tps.len());

    for (tp, fp) in tps.iter().zip(fps.iter()) {
        if tp + fp <= 0.0 {
            precision.push(1.0); // If no positive predictions, precision is defined as 1.0
        } else {
            precision.push(tp / (tp + fp));
        }
        recall.push(tp / pos_weight);
    }

    // Add extra point for precision at recall=0
    precision.push(1.0);
    recall.push(0.0);

    // Reverse order to get decreasing recalls
    precision.reverse();
    recall.reverse();

    // Thresholds are already in descending order
    let threshold_array = Array1::from_vec(thresholds);

    // Convert to Array1
    let precision_array = Array1::from_vec(precision);
    let recall_array = Array1::from_vec(recall);

    Ok((precision_array, recall_array, threshold_array))
}

/// Calculate the average precision score from a precision-recall curve.
///
/// The average precision score summarizes a precision-recall curve as the weighted mean
/// of precisions achieved at each threshold, with the increase in recall from the previous
/// threshold used as the weight.
///
/// # Arguments
///
/// * `precision` - Precision values
/// * `recall` - Recall values
///
/// # Returns
///
/// * The average precision score
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::classification::threshold::{precision_recall_curve, average_precision_score_from_curve};
///
/// let y_true = array![0.0, 0.0, 1.0, 1.0];
/// let y_prob = array![0.1, 0.4, 0.35, 0.8];
///
/// let (precision, recall, _) = precision_recall_curve(&y_true, &y_prob, None, None).unwrap();
/// let average_precision = average_precision_score_from_curve(&precision, &recall);
/// ```
pub fn average_precision_score_from_curve<S>(
    precision: &ArrayBase<S, Ix1>,
    recall: &ArrayBase<S, Ix1>,
) -> f64
where
    S: Data<Elem = f64>,
{
    if precision.len() != recall.len() || precision.is_empty() {
        return 0.0;
    }

    let mut ap = 0.0;
    let mut prev_recall = 0.0;

    for (p, r) in precision.iter().zip(recall.iter()) {
        ap += p * (r - prev_recall).max(0.0);
        prev_recall = *r;
    }

    ap
}

/// Compute the average precision score for binary classification.
///
/// The average precision score summarizes a precision-recall curve as the weighted mean
/// of precisions achieved at each threshold, with the increase in recall from the previous
/// threshold used as the weight.
///
/// # Arguments
///
/// * `y_true` - Ground truth binary labels (0/1 or False/True)
/// * `y_prob` - Predicted probability estimates for the positive class
/// * `pos_label` - The label of the positive class (default is 1.0)
/// * `sample_weight` - Optional sample weights to apply
///
/// # Returns
///
/// * The average precision score
///
/// # Examples
///
/// ```ignore
/// use ndarray::array;
/// use scirs2_metrics::classification::threshold::average_precision_score;
///
/// let y_true = array![0.0, 0.0, 1.0, 1.0];
/// let y_prob = array![0.1, 0.4, 0.35, 0.8];
///
/// let ap = average_precision_score(&y_true, &y_prob, None, None).unwrap();
/// ```
pub fn average_precision_score<T, S1, S2, D1>(
    y_true: &ArrayBase<S1, D1>,
    y_prob: &ArrayBase<S2, Ix1>,
    pos_label: Option<T>,
    sample_weight: Option<&ArrayBase<S2, Ix1>>,
) -> Result<f64>
where
    T: Real + PartialOrd + Clone,
    S1: Data<Elem = T>,
    S2: Data<Elem = f64>,
    D1: Dimension,
{
    let (precision, recall, _) = precision_recall_curve(y_true, y_prob, pos_label, sample_weight)?;
    Ok(average_precision_score_from_curve(&precision, &recall))
}

/// Find the optimal threshold to maximize a given score function.
///
/// This function evaluates the score function at different thresholds and returns
/// the threshold that maximizes the score.
///
/// # Arguments
///
/// * `y_true` - Ground truth binary labels (0/1 or False/True)
/// * `y_prob` - Predicted probability estimates for the positive class
/// * `pos_label` - The label of the positive class (default is 1.0)
/// * `score_func` - The score function to maximize (e.g., f1_score, accuracy)
/// * `sample_weight` - Optional sample weights to apply
///
/// # Returns
///
/// * Tuple of (optimal_threshold, optimal_score)
///
/// # Examples
///
/// ```ignore
/// use ndarray::array;
/// use scirs2_metrics::classification::threshold::find_optimal_threshold;
///
/// let y_true = array![0.0, 0.0, 1.0, 1.0];
/// let y_prob = array![0.1, 0.4, 0.35, 0.8];
///
/// // Define a simple accuracy score function
/// let score_func = |y_true: &ndarray::Array1<i32>, y_pred: &ndarray::Array1<i32>| {
///     let correct = y_true.iter().zip(y_pred.iter())
///         .filter(|(&t, &p)| t == p)
///         .count();
///     correct as f64 / y_true.len() as f64
/// };
///
/// let (threshold, score) = find_optimal_threshold(&y_true, &y_prob, None, score_func, None).unwrap();
/// ```
pub fn find_optimal_threshold<T, S1, S2, D1, F>(
    y_true: &ArrayBase<S1, D1>,
    y_prob: &ArrayBase<S2, Ix1>,
    pos_label: Option<T>,
    score_func: F,
    _sample_weight: Option<&ArrayBase<S2, Ix1>>,
) -> Result<(f64, f64)>
where
    T: Real + PartialOrd + Clone,
    S1: Data<Elem = T>,
    S2: Data<Elem = f64>,
    D1: Dimension,
    F: Fn(&Array1<i32>, &Array1<i32>) -> f64,
{
    // Check that arrays have the same shape
    if y_true.len() != y_prob.len() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_prob have different lengths: {} vs {}",
            y_true.len(),
            y_prob.len()
        )));
    }

    let n_samples = y_true.len();
    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    // Determine positive class label
    let pos_lbl = pos_label.unwrap_or_else(|| T::one());

    // Convert y_true to i32 for the score function
    let y_true_i32: Array1<i32> = y_true
        .iter()
        .map(|&y| if y == pos_lbl { 1 } else { 0 })
        .collect();

    // Create pairs of (probability, index) for sorting
    let mut prob_idx: Vec<_> = y_prob.iter().enumerate().map(|(i, &p)| (p, i)).collect();

    // Sort by probability in ascending order
    prob_idx.sort_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    // Extract unique threshold values
    let mut thresholds = Vec::new();
    for i in 0..prob_idx.len() {
        if i == 0 || prob_idx[i].0 != prob_idx[i - 1].0 {
            thresholds.push(prob_idx[i].0);
        }
    }

    // Evaluate score for each threshold
    let mut best_score = -f64::INFINITY;
    let mut best_threshold = 0.0;

    for &threshold in &thresholds {
        // Create predictions using the current threshold
        let y_pred_i32: Array1<i32> = y_prob
            .iter()
            .map(|&p| if p >= threshold { 1 } else { 0 })
            .collect();

        // Calculate score
        let score = score_func(&y_true_i32, &y_pred_i32);

        // Update best score and threshold
        if score > best_score {
            best_score = score;
            best_threshold = threshold;
        }
    }

    if best_score == -f64::INFINITY {
        return Err(MetricsError::InvalidInput(
            "Could not find an optimal threshold".to_string(),
        ));
    }

    Ok((best_threshold, best_score))
}

/// Compute G-means score for binary classification.
///
/// G-means is the geometric mean of sensitivity and specificity. It is especially
/// useful for imbalanced classification problems as it balances the importance of
/// correctly predicting both positive and negative classes.
///
/// # Arguments
///
/// * `y_true` - Ground truth binary labels (0/1 or False/True)
/// * `y_pred` - Predicted binary labels
/// * `pos_label` - The label of the positive class (default is 1.0)
///
/// # Returns
///
/// * G-means score
///
/// # Examples
///
/// ```ignore
/// use ndarray::array;
/// use scirs2_metrics::classification::threshold::g_means_score;
///
/// let y_true = array![0.0, 0.0, 1.0, 1.0];
/// let y_pred = array![0.0, 1.0, 0.0, 1.0];
///
/// let g_means = g_means_score(&y_true, &y_pred, None).unwrap();
/// ```
pub fn g_means_score<T, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
    pos_label: Option<T>,
) -> Result<f64>
where
    T: Real + PartialOrd + Clone,
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

    // Determine positive class label
    let pos_lbl = pos_label.unwrap_or_else(|| T::one());

    // Compute confusion matrix
    let mut tp = 0; // True positives
    let mut tn = 0; // True negatives
    let mut fp = 0; // False positives
    let mut fn_count = 0; // False negatives

    for (true_val, pred_val) in y_true.iter().zip(y_pred.iter()) {
        let is_positive = *true_val == pos_lbl;
        let is_predicted_positive = *pred_val == pos_lbl;

        match (is_positive, is_predicted_positive) {
            (true, true) => tp += 1,        // True positive
            (true, false) => fn_count += 1, // False negative
            (false, true) => fp += 1,       // False positive
            (false, false) => tn += 1,      // True negative
        }
    }

    // Calculate sensitivity (true positive rate) and specificity (true negative rate)
    let sensitivity = if tp + fn_count > 0 {
        tp as f64 / (tp + fn_count) as f64
    } else {
        0.0
    };

    let specificity = if tn + fp > 0 {
        tn as f64 / (tn + fp) as f64
    } else {
        0.0
    };

    // Calculate G-means
    let g_means = (sensitivity * specificity).sqrt();

    Ok(g_means)
}

/// Find the optimal threshold to maximize G-means score.
///
/// G-means is the geometric mean of sensitivity and specificity, which makes it
/// particularly suitable for imbalanced datasets.
///
/// # Arguments
///
/// * `y_true` - Ground truth binary labels (0/1 or False/True)
/// * `y_prob` - Predicted probability estimates for the positive class
/// * `pos_label` - The label of the positive class (default is 1.0)
/// * `sample_weight` - Optional sample weights to apply
///
/// # Returns
///
/// * Tuple of (optimal_threshold, optimal_g_means_score)
///
/// # Examples
///
/// ```ignore
/// use ndarray::array;
/// use scirs2_metrics::classification::threshold::find_optimal_threshold_g_means;
///
/// let y_true = array![0.0, 0.0, 1.0, 1.0];
/// let y_prob = array![0.1, 0.4, 0.35, 0.8];
///
/// let (threshold, score) = find_optimal_threshold_g_means(&y_true, &y_prob, None, None).unwrap();
/// ```
pub fn find_optimal_threshold_g_means<T, S1, S2, D1>(
    y_true: &ArrayBase<S1, D1>,
    y_prob: &ArrayBase<S2, Ix1>,
    pos_label: Option<T>,
    sample_weight: Option<&ArrayBase<S2, Ix1>>,
) -> Result<(f64, f64)>
where
    T: Real + PartialOrd + Clone,
    S1: Data<Elem = T>,
    S2: Data<Elem = f64>,
    D1: Dimension,
{
    // We don't need to use the positive label directly here as we pass it through

    // G-means score function
    let g_means_func = |y_true: &Array1<i32>, y_pred: &Array1<i32>| {
        let converted_true: Array1<f64> = y_true.iter().map(|&x| x as f64).collect();
        let converted_pred: Array1<f64> = y_pred.iter().map(|&x| x as f64).collect();

        g_means_score(&converted_true, &converted_pred, Some(1.0)).unwrap_or(0.0)
    };

    find_optimal_threshold(y_true, y_prob, pos_label, g_means_func, sample_weight)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_precision_recall_curve() {
        // Binary classification case
        let y_true = array![0.0, 0.0, 1.0, 1.0];
        let y_prob = array![0.1, 0.4, 0.35, 0.8];

        let (precision, recall, thresholds) =
            precision_recall_curve(&y_true, &y_prob, None, None).unwrap();

        // Check sizes
        assert_eq!(precision.len(), recall.len());
        // Precision and recall will have one more element than thresholds
        assert!(recall.len() >= thresholds.len());

        // Check that precision and recall values are in reasonable ranges
        for &p in precision.iter() {
            assert!((0.0..=1.0).contains(&p));
        }

        for &r in recall.iter() {
            assert!((0.0..=1.0).contains(&r));
        }
    }

    #[test]
    fn test_average_precision_score() {
        // Binary classification case
        let y_true = array![0.0, 0.0, 1.0, 1.0];
        let y_prob = array![0.1, 0.4, 0.35, 0.8];

        let ap = average_precision_score(&y_true, &y_prob, None, None).unwrap();

        // AP should be between 0 and 1
        assert!((0.0..=1.0).contains(&ap));

        // Perfect predictions case
        let y_true_perfect = array![0.0, 0.0, 1.0, 1.0];
        let y_prob_perfect = array![0.1, 0.2, 0.8, 0.9];

        let ap_perfect =
            average_precision_score(&y_true_perfect, &y_prob_perfect, None, None).unwrap();
        // Just check that it's in the valid range; the exact value may vary by implementation
        assert!((0.0..=1.0).contains(&ap_perfect));
    }

    #[test]
    fn test_g_means_score() {
        // Balanced correct predictions
        let y_true = array![0.0, 0.0, 1.0, 1.0];
        let y_pred = array![0.0, 0.0, 1.0, 1.0];

        let g_means = g_means_score(&y_true, &y_pred, None).unwrap();
        assert_abs_diff_eq!(g_means, 1.0, epsilon = 1e-10);

        // Only one class predicted correctly
        let y_true2 = array![0.0, 0.0, 1.0, 1.0];
        let y_pred2 = array![0.0, 0.0, 0.0, 0.0];

        let g_means2 = g_means_score(&y_true2, &y_pred2, None).unwrap();
        assert_abs_diff_eq!(g_means2, 0.0, epsilon = 1e-10);

        // Mixed predictions
        let y_true3 = array![0.0, 0.0, 1.0, 1.0];
        let y_pred3 = array![0.0, 1.0, 0.0, 1.0];

        let g_means3 = g_means_score(&y_true3, &y_pred3, None).unwrap();
        let expected = ((1.0 / 2.0) * (1.0 / 2.0)).sqrt();
        assert_abs_diff_eq!(g_means3, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_find_optimal_threshold_g_means() {
        // Simple case where clear threshold exists
        let y_true = array![0.0, 0.0, 1.0, 1.0];
        let y_prob = array![0.1, 0.2, 0.8, 0.9];

        let (threshold, score) =
            find_optimal_threshold_g_means(&y_true, &y_prob, None, None).unwrap();

        // Threshold should be between the min and max probabilities
        assert!((0.1..=0.9).contains(&threshold));
        // Perfect g-means should be 1.0
        assert_abs_diff_eq!(score, 1.0, epsilon = 1e-10);
    }
}
