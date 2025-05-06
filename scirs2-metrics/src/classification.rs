//! Classification metrics module
//!
//! This module provides functions for evaluating classification models, including
//! accuracy, precision, recall, F1 score, ROC AUC, and advanced metrics.
//!
//! ## Basic Metrics
//!
//! Basic classification metrics include accuracy, precision, recall, and F1 score.
//!
//! ## Advanced Metrics
//!
//! Advanced metrics include Matthews Correlation Coefficient, balanced accuracy,
//! Cohen's kappa, Brier score, Jaccard similarity, and Hamming loss.
//!
//! ```
//! use ndarray::array;
//! use scirs2_metrics::classification::advanced::{matthews_corrcoef, balanced_accuracy_score};
//!
//! let y_true = array![0, 1, 2, 0, 1, 2];
//! let y_pred = array![0, 2, 1, 0, 0, 2];
//!
//! let mcc = matthews_corrcoef(&y_true, &y_pred).unwrap();
//! let bal_acc = balanced_accuracy_score(&y_true, &y_pred).unwrap();
//! ```

pub mod advanced;

use ndarray::{Array1, Array2, ArrayBase, Data, Dimension};
use num_traits::NumCast;

use crate::error::{MetricsError, Result};

/// Calculates accuracy score, the fraction of correctly classified samples
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) labels
/// * `y_pred` - Predicted labels, as returned by a classifier
///
/// # Returns
///
/// * The fraction of correctly classified samples (float)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::classification::accuracy_score;
///
/// let y_true = array![0, 1, 2, 3];
/// let y_pred = array![0, 2, 1, 3];
///
/// let acc = accuracy_score(&y_true, &y_pred).unwrap();
/// assert!((acc - 0.5).abs() < 1e-10); // 2 out of 4 are correct
/// ```
pub fn accuracy_score<T, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
) -> Result<f64>
where
    T: PartialEq + NumCast + Clone,
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

    // Count correct predictions
    let mut n_correct = 0;
    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        if yt == yp {
            n_correct += 1;
        }
    }

    Ok(n_correct as f64 / n_samples as f64)
}

/// Calculates a confusion matrix to evaluate the accuracy of a classification
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) labels
/// * `y_pred` - Predicted labels, as returned by a classifier
/// * `labels` - Optional list of label values to index the matrix. This may be
///   used to reorder or select a subset of labels. If None, those that appear
///   at least once in y_true or y_pred are used in sorted order.
///
/// # Returns
///
/// * Confusion matrix (Array2<u64>)
/// * Vector of classes in order (Array1<T>)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::classification::confusion_matrix;
///
/// let y_true = array![0, 1, 2, 0, 1, 2];
/// let y_pred = array![0, 2, 1, 0, 0, 2];
///
/// let (cm, classes) = confusion_matrix(&y_true, &y_pred, None).unwrap();
/// // Expected confusion matrix:
/// // [[2, 0, 0],
/// //  [1, 0, 1],
/// //  [0, 1, 1]]
///
/// assert_eq!(cm[[0, 0]], 2); // True 0, predicted 0
/// assert_eq!(cm[[1, 0]], 1); // True 1, predicted 0
/// assert_eq!(cm[[1, 2]], 1); // True 1, predicted 2
/// assert_eq!(cm[[2, 1]], 1); // True 2, predicted 1
/// assert_eq!(cm[[2, 2]], 1); // True 2, predicted 2
/// ```
pub fn confusion_matrix<T, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
    labels: Option<&[T]>,
) -> Result<(Array2<u64>, Array1<T>)>
where
    T: PartialEq + NumCast + Clone + Ord + std::hash::Hash + std::fmt::Debug,
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

    // Determine the classes
    let classes = if let Some(labels) = labels {
        let mut cls = Vec::with_capacity(labels.len());
        for label in labels {
            cls.push(label.clone());
        }
        cls
    } else {
        let mut cls = std::collections::BTreeSet::new();
        for yt in y_true.iter() {
            cls.insert(yt.clone());
        }
        for yp in y_pred.iter() {
            cls.insert(yp.clone());
        }
        cls.into_iter().collect()
    };

    // Create the confusion matrix
    let n_classes = classes.len();
    let mut cm = Array2::zeros((n_classes, n_classes));

    // Create a map from class to index
    let mut class_to_idx = std::collections::HashMap::new();
    for (i, c) in classes.iter().enumerate() {
        class_to_idx.insert(c, i);
    }

    // Fill the confusion matrix
    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        if let (Some(&i), Some(&j)) = (class_to_idx.get(yt), class_to_idx.get(yp)) {
            cm[[i, j]] += 1;
        }
    }

    Ok((cm, Array1::from(classes)))
}

/// Calculates the precision score for binary classification
///
/// The precision is the ratio `tp / (tp + fp)` where `tp` is the number of
/// true positives and `fp` the number of false positives. The precision is
/// intuitively the ability of the classifier not to label as positive a sample
/// that is negative.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) binary labels
/// * `y_pred` - Predicted binary labels, as returned by a classifier
/// * `pos_label` - The label to report as positive class
///
/// # Returns
///
/// * The precision score (float between 0.0 and 1.0)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::classification::precision_score;
///
/// let y_true = array![0, 1, 0, 0, 1, 1];
/// let y_pred = array![0, 0, 1, 0, 1, 1];
///
/// let precision = precision_score(&y_true, &y_pred, 1).unwrap();
/// // There are 2 true positives and 1 false positive
/// assert!((precision - 2.0/3.0).abs() < 1e-10);
/// ```
pub fn precision_score<T, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
    pos_label: T,
) -> Result<f64>
where
    T: PartialEq + NumCast + Clone,
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

    // Count true positives and false positives
    let mut true_positives = 0;
    let mut false_positives = 0;

    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        if yp == &pos_label {
            if yt == yp {
                true_positives += 1;
            } else {
                false_positives += 1;
            }
        }
    }

    // Calculate precision
    if true_positives + false_positives == 0 {
        Ok(0.0) // No positive predictions, precision is 0
    } else {
        Ok(true_positives as f64 / (true_positives + false_positives) as f64)
    }
}

/// Calculates the recall score for binary classification
///
/// The recall is the ratio `tp / (tp + fn)` where `tp` is the number of
/// true positives and `fn` the number of false negatives. The recall is
/// intuitively the ability of the classifier to find all the positive samples.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) binary labels
/// * `y_pred` - Predicted binary labels, as returned by a classifier
/// * `pos_label` - The label to report as positive class
///
/// # Returns
///
/// * The recall score (float between 0.0 and 1.0)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::classification::recall_score;
///
/// let y_true = array![0, 1, 0, 0, 1, 1];
/// let y_pred = array![0, 0, 1, 0, 1, 1];
///
/// let recall = recall_score(&y_true, &y_pred, 1).unwrap();
/// // There are 2 true positives and 1 false negative
/// assert!((recall - 2.0/3.0).abs() < 1e-10);
/// ```
pub fn recall_score<T, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
    pos_label: T,
) -> Result<f64>
where
    T: PartialEq + NumCast + Clone,
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

    // Count true positives and false negatives
    let mut true_positives = 0;
    let mut false_negatives = 0;

    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        if yt == &pos_label {
            if yp == yt {
                true_positives += 1;
            } else {
                false_negatives += 1;
            }
        }
    }

    // Calculate recall
    if true_positives + false_negatives == 0 {
        Ok(0.0) // No actual positives, recall is 0
    } else {
        Ok(true_positives as f64 / (true_positives + false_negatives) as f64)
    }
}

/// Calculates the F1 score for binary classification
///
/// The F1 score is the harmonic mean of precision and recall:
/// `F1 = 2 * (precision * recall) / (precision + recall)`
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) binary labels
/// * `y_pred` - Predicted binary labels, as returned by a classifier
/// * `pos_label` - The label to report as positive class
///
/// # Returns
///
/// * The F1 score (float between 0.0 and 1.0)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::classification::f1_score;
///
/// let y_true = array![0, 1, 0, 0, 1, 1];
/// let y_pred = array![0, 0, 1, 0, 1, 1];
///
/// let f1 = f1_score(&y_true, &y_pred, 1).unwrap();
/// ```
pub fn f1_score<T, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
    pos_label: T,
) -> Result<f64>
where
    T: PartialEq + NumCast + Clone,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    D1: Dimension,
    D2: Dimension,
{
    let precision = precision_score(y_true, y_pred, pos_label.clone())?;
    let recall = recall_score(y_true, y_pred, pos_label)?;

    if precision + recall == 0.0 {
        return Ok(0.0);
    }

    Ok(2.0 * precision * recall / (precision + recall))
}

/// Calculate binary log loss, also known as binary cross-entropy
///
/// This is the loss function used in (multinomial) logistic regression
/// and neural networks (with softmax output).
///
/// # Arguments
///
/// * `y_true` - Ground truth binary labels
/// * `y_prob` - Predicted probabilities for the positive class
/// * `eps` - Small value to avoid log(0)
///
/// # Returns
///
/// * The log loss (float)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::classification::binary_log_loss;
///
/// let y_true = array![0, 1, 1, 0];
/// let y_prob = array![0.1, 0.9, 0.8, 0.3];
///
/// let loss = binary_log_loss(&y_true, &y_prob, 1e-15).unwrap();
/// ```
pub fn binary_log_loss<S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_prob: &ArrayBase<S2, D2>,
    eps: f64,
) -> Result<f64>
where
    S1: Data<Elem = u32>,
    S2: Data<Elem = f64>,
    D1: Dimension,
    D2: Dimension,
{
    // Check that arrays have the same shape
    if y_true.shape() != y_prob.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_prob have different shapes: {:?} vs {:?}",
            y_true.shape(),
            y_prob.shape()
        )));
    }

    let n_samples = y_true.len();
    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    // Compute log loss
    let mut loss = 0.0;
    for (yt, yp) in y_true.iter().zip(y_prob.iter()) {
        // Clip probability to avoid log(0)
        let clipped_yp = yp.max(eps).min(1.0 - eps);

        if *yt == 1 {
            loss -= (clipped_yp).ln();
        } else {
            loss -= (1.0 - clipped_yp).ln();
        }
    }

    Ok(loss / n_samples as f64)
}

/// Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)
///
/// The ROC curve is created by plotting the true positive rate (TPR) against
/// the false positive rate (FPR) at various threshold settings.
///
/// # Arguments
///
/// * `y_true` - Ground truth binary labels
/// * `y_score` - Target scores (can be probability estimates of the positive class)
///
/// # Returns
///
/// * The ROC AUC score (float between 0.0 and 1.0)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::classification::roc_auc_score;
///
/// let y_true = array![0, 0, 1, 1];
/// let y_score = array![0.1, 0.4, 0.35, 0.8];
///
/// let auc = roc_auc_score(&y_true, &y_score).unwrap();
/// ```
pub fn roc_auc_score<S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_score: &ArrayBase<S2, D2>,
) -> Result<f64>
where
    S1: Data<Elem = u32>,
    S2: Data<Elem = f64>,
    D1: Dimension,
    D2: Dimension,
{
    // Check that arrays have the same shape
    if y_true.shape() != y_score.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_score have different shapes: {:?} vs {:?}",
            y_true.shape(),
            y_score.shape()
        )));
    }

    let n_samples = y_true.len();
    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    // Compute the number of positive and negative samples
    let mut n_pos = 0;
    let mut n_neg = 0;
    for &yt in y_true.iter() {
        if yt == 1 {
            n_pos += 1;
        } else {
            n_neg += 1;
        }
    }

    if n_pos == 0 || n_neg == 0 {
        return Err(MetricsError::InvalidInput(
            "ROC AUC score is not defined when only one class is present".to_string(),
        ));
    }

    // Collect scores and true labels
    let mut scores_and_labels = Vec::with_capacity(n_samples);
    for (yt, ys) in y_true.iter().zip(y_score.iter()) {
        scores_and_labels.push((*ys, *yt));
    }

    // Sort scores in descending order
    scores_and_labels.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Compute AUC by summing up the trapezoids
    let mut auc = 0.0;
    let mut false_positive = 0;
    let mut true_positive = 0;
    let mut last_false_positive = 0;
    let mut last_true_positive = 0;
    let mut last_score = f64::INFINITY;

    for (score, label) in scores_and_labels {
        if score != last_score {
            // Add the area of the trapezoid
            auc += (false_positive - last_false_positive) as f64
                * (true_positive + last_true_positive) as f64
                / 2.0;
            last_score = score;
            last_false_positive = false_positive;
            last_true_positive = true_positive;
        }

        if label == 1 {
            true_positive += 1;
        } else {
            false_positive += 1;
        }
    }

    // Add the final trapezoid
    auc += (n_neg - last_false_positive) as f64 * (true_positive + last_true_positive) as f64 / 2.0;

    // Normalize
    auc /= (n_pos * n_neg) as f64;

    Ok(auc)
}

/// Generates a text report showing the main classification metrics
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated targets as returned by a classifier
/// * `labels` - Optional list of label values to include in the report
///
/// # Returns
///
/// * A string containing the classification report
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::classification::classification_report;
///
/// let y_true = array![0, 1, 2, 0, 1, 2];
/// let y_pred = array![0, 2, 1, 0, 0, 2];
///
/// let report = classification_report(&y_true, &y_pred, None).unwrap();
/// println!("{}", report);
/// ```
pub fn classification_report<T, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
    labels: Option<&[T]>,
) -> Result<String>
where
    T: PartialEq + NumCast + Clone + Ord + std::hash::Hash + std::fmt::Debug,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    D1: Dimension,
    D2: Dimension,
{
    // Get confusion matrix
    let (cm, classes) = confusion_matrix(y_true, y_pred, labels)?;

    // Prepare report
    let mut report = String::new();
    report.push_str("              precision    recall  f1-score   support\n\n");

    let n_classes = classes.len();
    let mut total_precision = 0.0;
    let mut total_recall = 0.0;
    let mut total_f1 = 0.0;
    let mut total_support = 0;

    for i in 0..n_classes {
        let class_label = format!("{:?}", classes[i]);

        // Calculate metrics for this class
        let true_positives = cm[[i, i]];
        let false_positives = cm.column(i).sum() - true_positives;
        let false_negatives = cm.row(i).sum() - true_positives;
        let support = cm.row(i).sum();

        let precision = if true_positives + false_positives == 0 {
            0.0
        } else {
            true_positives as f64 / (true_positives + false_positives) as f64
        };

        let recall = if true_positives + false_negatives == 0 {
            0.0
        } else {
            true_positives as f64 / (true_positives + false_negatives) as f64
        };

        let f1 = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };

        // Add to totals
        total_precision += precision;
        total_recall += recall;
        total_f1 += f1;
        total_support += support as usize;

        // Add line to report
        report.push_str(&format!(
            "{:>14} {:9.2} {:9.2} {:9.2} {:9}\n",
            class_label, precision, recall, f1, support
        ));
    }

    report.push('\n');

    // Calculate averages
    let avg_precision = total_precision / n_classes as f64;
    let avg_recall = total_recall / n_classes as f64;
    let avg_f1 = total_f1 / n_classes as f64;

    // Add averages to report
    report.push_str(&format!(
        "    avg / total {:9.2} {:9.2} {:9.2} {:9}\n",
        avg_precision, avg_recall, avg_f1, total_support
    ));

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_accuracy_score() {
        let y_true = array![0, 1, 2, 3];
        let y_pred = array![0, 2, 1, 3];

        let acc = accuracy_score(&y_true, &y_pred).unwrap();
        assert_abs_diff_eq!(acc, 0.5, epsilon = 1e-10); // 2 out of 4 are correct
    }

    #[test]
    fn test_confusion_matrix() {
        let y_true = array![0, 1, 2, 0, 1, 2];
        let y_pred = array![0, 2, 1, 0, 0, 2];

        let (cm, classes) = confusion_matrix(&y_true, &y_pred, None).unwrap();

        assert_eq!(cm.shape(), &[3, 3]);
        assert_eq!(classes.len(), 3);

        // Expected confusion matrix:
        // [[2, 0, 0],
        //  [1, 0, 1],
        //  [0, 1, 1]]
        assert_eq!(cm[[0, 0]], 2); // True 0, predicted 0
        assert_eq!(cm[[1, 0]], 1); // True 1, predicted 0
        assert_eq!(cm[[1, 2]], 1); // True 1, predicted 2
        assert_eq!(cm[[2, 1]], 1); // True 2, predicted 1
        assert_eq!(cm[[2, 2]], 1); // True 2, predicted 2
    }

    #[test]
    fn test_precision_recall_f1() {
        let y_true = array![0, 1, 0, 0, 1, 1];
        let y_pred = array![0, 0, 1, 0, 1, 1];

        let precision = precision_score(&y_true, &y_pred, 1).unwrap();
        let recall = recall_score(&y_true, &y_pred, 1).unwrap();
        let f1 = f1_score(&y_true, &y_pred, 1).unwrap();

        // precision = 2/3, recall = 2/3, f1 = 2*2/3*2/3 / (2/3 + 2/3) = 2/3
        assert_abs_diff_eq!(precision, 2.0 / 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(recall, 2.0 / 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(f1, 2.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_log_loss() {
        let y_true = array![0, 1, 1, 0];
        let y_prob = array![0.1, 0.9, 0.8, 0.3];

        let loss = binary_log_loss(&y_true, &y_prob, 1e-15).unwrap();
        // Expected loss: -[(log(0.9) + log(0.8) + log(0.9) + log(0.7))/4]
        let expected =
            -(((1.0_f64 - 0.1).ln() + 0.9_f64.ln() + 0.8_f64.ln() + (1.0_f64 - 0.3).ln()) / 4.0);
        assert_abs_diff_eq!(loss, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_roc_auc() {
        // Perfect separation
        let y_true = array![0, 0, 1, 1];
        let y_score = array![0.1, 0.2, 0.8, 0.9];
        let auc = roc_auc_score(&y_true, &y_score).unwrap();
        assert_abs_diff_eq!(auc, 1.0, epsilon = 1e-10);

        // Random
        let y_true = array![0, 1, 0, 1];
        let y_score = array![0.5, 0.5, 0.5, 0.5];
        let auc = roc_auc_score(&y_true, &y_score).unwrap();
        assert_abs_diff_eq!(auc, 0.5, epsilon = 1e-10);
    }
}
