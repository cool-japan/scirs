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
//!
//! ## One-vs-One Metrics
//!
//! One-vs-One metrics are useful for evaluating multi-class classification problems by
//! considering each pair of classes separately.
//!
//! ```
//! use ndarray::array;
//! use scirs2_metrics::classification::one_vs_one::{one_vs_one_accuracy, one_vs_one_f1_score};
//!
//! let y_true = array![0, 1, 2, 0, 1, 2];
//! let y_pred = array![0, 2, 1, 0, 0, 2];
//!
//! let ovo_acc = one_vs_one_accuracy(&y_true, &y_pred).unwrap();
//! let f1_scores = one_vs_one_f1_score(&y_true, &y_pred).unwrap();
//! ```

pub mod advanced;
pub mod curves;
pub mod one_vs_one;
pub mod threshold;
pub mod threshold_analyzer;

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
/// This is a special case of the F-beta score with beta=1.
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
    // F1 score is a special case of fbeta_score with beta = 1.0
    fbeta_score(y_true, y_pred, pos_label, 1.0)
}

/// Calculates the F-beta score for binary classification
///
/// The F-beta score is the weighted harmonic mean of precision and recall:
/// `F-beta = (1 + beta^2) * (precision * recall) / ((beta^2 * precision) + recall)`
///
/// The beta parameter determines the weight of recall in the combined score:
/// - beta < 1 gives more weight to precision
/// - beta > 1 gives more weight to recall
/// - beta = 1 gives equal weight to precision and recall (F1 score)
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) binary labels
/// * `y_pred` - Predicted binary labels, as returned by a classifier
/// * `pos_label` - The label to report as positive class
/// * `beta` - The weight of recall relative to precision (must be positive)
///
/// # Returns
///
/// * The F-beta score (float between 0.0 and 1.0)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::classification::fbeta_score;
///
/// let y_true = array![0, 1, 0, 0, 1, 1];
/// let y_pred = array![0, 0, 1, 0, 1, 1];
///
/// // F0.5 score (weighs precision higher than recall)
/// let f_half = fbeta_score(&y_true, &y_pred, 1, 0.5).unwrap();
///
/// // F2 score (weighs recall higher than precision)
/// let f_two = fbeta_score(&y_true, &y_pred, 1, 2.0).unwrap();
/// ```
pub fn fbeta_score<T, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
    pos_label: T,
    beta: f64,
) -> Result<f64>
where
    T: PartialEq + NumCast + Clone,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    D1: Dimension,
    D2: Dimension,
{
    if beta <= 0.0 {
        return Err(MetricsError::InvalidInput(format!(
            "beta must be positive, got {}",
            beta
        )));
    }

    let precision = precision_score(y_true, y_pred, pos_label.clone())?;
    let recall = recall_score(y_true, y_pred, pos_label)?;

    if precision + recall == 0.0 {
        return Ok(0.0);
    }

    let beta_squared = beta * beta;
    Ok((1.0 + beta_squared) * precision * recall / ((beta_squared * precision) + recall))
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

/// Computes the lift chart values for binary classification
///
/// The lift chart shows how much better a model performs compared to a random model.
/// It is particularly useful in marketing and customer targeting applications.
///
/// # Arguments
///
/// * `y_true` - Ground truth binary labels (0 or 1)
/// * `y_score` - Predicted probabilities for the positive class
/// * `n_bins` - Number of bins for the lift chart
///
/// # Returns
///
/// * A tuple containing three arrays:
///   * `percentiles` - The percentiles used (0-100)
///   * `lift_values` - The lift values for each percentile
///   * `cum_gains` - Cumulative gains values (for gain chart)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::classification::lift_chart;
///
/// let y_true = array![0, 0, 1, 0, 1, 1, 0, 1, 0, 1];
/// let y_score = array![0.1, 0.2, 0.7, 0.3, 0.8, 0.9, 0.4, 0.6, 0.2, 0.5];
///
/// let (percentiles, lift_values, cum_gains) = lift_chart(&y_true, &y_score, 10).unwrap();
/// ```
pub fn lift_chart<S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_score: &ArrayBase<S2, D2>,
    n_bins: usize,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>)>
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

    // Validate y_true contains only binary values
    for yt in y_true.iter() {
        if *yt != 0 && *yt != 1 {
            return Err(MetricsError::InvalidInput(
                "y_true must contain only binary values (0 or 1)".to_string(),
            ));
        }
    }

    // Validate n_bins
    if n_bins < 1 {
        return Err(MetricsError::InvalidInput(
            "n_bins must be at least 1".to_string(),
        ));
    }

    // Compute the overall positive rate (baseline)
    let n_positives = y_true.iter().filter(|&&y| y == 1).count();
    if n_positives == 0 || n_positives == n_samples {
        return Err(MetricsError::InvalidInput(
            "y_true must contain both positive and negative samples".to_string(),
        ));
    }
    let baseline_rate = n_positives as f64 / n_samples as f64;

    // Pair scores with true labels and sort by scores in descending order
    let mut paired_data: Vec<(f64, u32)> = y_score
        .iter()
        .zip(y_true.iter())
        .map(|(&score, &label)| (score, label))
        .collect();
    paired_data.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Calculate percentiles, lift values, and cumulative gains
    let bin_size = n_samples / n_bins;
    let mut percentiles = Vec::with_capacity(n_bins);
    let mut lift_values = Vec::with_capacity(n_bins);
    let mut cum_gains = Vec::with_capacity(n_bins);

    for i in 0..n_bins {
        // Calculate percentile
        let percentile = (i + 1) as f64 * 100.0 / n_bins as f64;

        // Calculate number of samples to consider (based on percentile)
        let n_considered = if i == n_bins - 1 {
            // Include all samples in the last bin
            n_samples
        } else {
            (i + 1) * bin_size
        };

        // Count positives in this subset
        let positives_in_bin = paired_data[0..n_considered]
            .iter()
            .filter(|(_, label)| *label == 1)
            .count();

        // Calculate lift and cumulative gain
        let bin_rate = positives_in_bin as f64 / n_considered as f64;
        let lift = bin_rate / baseline_rate;
        let cum_gain = positives_in_bin as f64 / n_positives as f64;

        percentiles.push(percentile);
        lift_values.push(lift);
        cum_gains.push(cum_gain);
    }

    Ok((
        Array1::from(percentiles),
        Array1::from(lift_values),
        Array1::from(cum_gains),
    ))
}

/// Computes the gain chart values for binary classification
///
/// The gain chart (or cumulative gains chart) shows the percentage of positive
/// outcomes captured at each percentile when observations are ranked by predicted probability.
///
/// # Arguments
///
/// * `y_true` - Ground truth binary labels (0 or 1)
/// * `y_score` - Predicted probabilities for the positive class
/// * `n_bins` - Number of bins for the gain chart
///
/// # Returns
///
/// * A tuple containing two arrays:
///   * `percentiles` - The percentiles used (0-100)
///   * `cum_gains` - Cumulative gains values at each percentile
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::classification::gain_chart;
///
/// let y_true = array![0, 0, 1, 0, 1, 1, 0, 1, 0, 1];
/// let y_score = array![0.1, 0.2, 0.7, 0.3, 0.8, 0.9, 0.4, 0.6, 0.2, 0.5];
///
/// let (percentiles, cum_gains) = gain_chart(&y_true, &y_score, 10).unwrap();
/// ```
pub fn gain_chart<S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_score: &ArrayBase<S2, D2>,
    n_bins: usize,
) -> Result<(Array1<f64>, Array1<f64>)>
where
    S1: Data<Elem = u32>,
    S2: Data<Elem = f64>,
    D1: Dimension,
    D2: Dimension,
{
    // Reuse lift_chart function to get the data
    let (percentiles, _, cum_gains) = lift_chart(y_true, y_score, n_bins)?;
    Ok((percentiles, cum_gains))
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
    fn test_fbeta_score() {
        let y_true = array![0, 1, 0, 0, 1, 1];
        let y_pred = array![0, 0, 1, 0, 1, 1];

        // F1 score (beta = 1.0)
        let f1 = fbeta_score(&y_true, &y_pred, 1, 1.0).unwrap();
        assert_abs_diff_eq!(f1, 2.0 / 3.0, epsilon = 1e-10);

        // F0.5 score (weighs precision higher than recall)
        let f_half = fbeta_score(&y_true, &y_pred, 1, 0.5).unwrap();
        // With beta=0.5, beta²=0.25
        // F0.5 = (1+0.25) * 2/3 * 2/3 / (0.25*2/3 + 2/3) = 1.25 * 4/9 / (1/6 + 2/3) = 5/9 / 5/6 = 5/9 * 6/5 = 30/45 = 2/3
        assert_abs_diff_eq!(f_half, 2.0 / 3.0, epsilon = 1e-10);

        // F2 score (weighs recall higher than precision)
        let f_two = fbeta_score(&y_true, &y_pred, 1, 2.0).unwrap();
        // With beta=2.0, beta²=4.0
        // F2 = (1+4) * 2/3 * 2/3 / (4*2/3 + 2/3) = 5 * 4/9 / (8/3 + 2/3) = 20/9 / 10/3 = 20/9 * 3/10 = 60/90 = 2/3
        assert_abs_diff_eq!(f_two, 2.0 / 3.0, epsilon = 1e-10);

        // This example has equal precision and recall, so all F-beta scores are the same
        // Let's try a more interesting example with different precision and recall
        let y_true = array![1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
        let y_pred = array![1, 1, 1, 0, 0, 0, 0, 0, 1, 1];
        // precision = 3/5 = 0.6, recall = 3/5 = 0.6

        // F1 score (beta = 1.0)
        let f1 = fbeta_score(&y_true, &y_pred, 1, 1.0).unwrap();
        assert_abs_diff_eq!(f1, 0.6, epsilon = 1e-10);

        // F0.5 score (weighs precision higher than recall)
        let f_half = fbeta_score(&y_true, &y_pred, 1, 0.5).unwrap();
        // With beta=0.5, beta²=0.25
        // F0.5 = (1+0.25) * 0.6 * 0.6 / (0.25*0.6 + 0.6) = 1.25 * 0.36 / (0.15 + 0.6) = 0.45 / 0.75 = 0.6
        assert_abs_diff_eq!(f_half, 0.6, epsilon = 1e-10);

        // F2 score (weighs recall higher than precision)
        let f_two = fbeta_score(&y_true, &y_pred, 1, 2.0).unwrap();
        // With beta=2.0, beta²=4.0
        // F2 = (1+4) * 0.6 * 0.6 / (4*0.6 + 0.6) = 5 * 0.36 / (2.4 + 0.6) = 1.8 / 3.0 = 0.6
        assert_abs_diff_eq!(f_two, 0.6, epsilon = 1e-10);

        // Let's try one more with different precision and recall
        let y_true = array![1, 1, 1, 1, 0, 0, 0, 0];
        let y_pred = array![1, 1, 0, 0, 0, 0, 1, 1];
        // precision = 2/4 = 0.5, recall = 2/4 = 0.5

        // F0.5 score (weighs precision higher than recall)
        let f_half = fbeta_score(&y_true, &y_pred, 1, 0.5).unwrap();
        // With beta=0.5, beta²=0.25
        // F0.5 = (1+0.25) * 0.5 * 0.5 / (0.25*0.5 + 0.5) = 1.25 * 0.25 / (0.125 + 0.5) = 0.3125 / 0.625 = 0.5
        assert_abs_diff_eq!(f_half, 0.5, epsilon = 1e-10);
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

    #[test]
    fn test_lift_chart() {
        let y_true = array![0, 0, 1, 0, 1, 1, 0, 1, 0, 1];
        let y_score = array![0.1, 0.2, 0.7, 0.3, 0.8, 0.9, 0.4, 0.6, 0.2, 0.5];

        // Test with 5 bins
        let (percentiles, lift_values, cum_gains) = lift_chart(&y_true, &y_score, 5).unwrap();

        // Verify dimensions
        assert_eq!(percentiles.len(), 5);
        assert_eq!(lift_values.len(), 5);
        assert_eq!(cum_gains.len(), 5);

        // Verify percentiles
        assert_abs_diff_eq!(percentiles[0], 20.0, epsilon = 1e-10);
        assert_abs_diff_eq!(percentiles[4], 100.0, epsilon = 1e-10);

        // The first 20% contains the highest scored cases, which should be mostly positive
        // This should give a lift value higher than 1
        assert!(lift_values[0] > 1.0);

        // The cumulative gains at 100% should be 1.0 (all positives)
        assert_abs_diff_eq!(cum_gains[4], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gain_chart() {
        let y_true = array![0, 0, 1, 0, 1, 1, 0, 1, 0, 1];
        let y_score = array![0.1, 0.2, 0.7, 0.3, 0.8, 0.9, 0.4, 0.6, 0.2, 0.5];

        // Test with 5 bins
        let (percentiles, cum_gains) = gain_chart(&y_true, &y_score, 5).unwrap();

        // Verify dimensions
        assert_eq!(percentiles.len(), 5);
        assert_eq!(cum_gains.len(), 5);

        // Verify percentiles
        assert_abs_diff_eq!(percentiles[0], 20.0, epsilon = 1e-10);
        assert_abs_diff_eq!(percentiles[4], 100.0, epsilon = 1e-10);

        // Cumulative gains should be non-decreasing
        for i in 1..cum_gains.len() {
            assert!(cum_gains[i] >= cum_gains[i - 1]);
        }

        // The cumulative gains at 100% should be 1.0 (all positives)
        assert_abs_diff_eq!(cum_gains[4], 1.0, epsilon = 1e-10);
    }
}
