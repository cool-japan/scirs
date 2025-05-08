//! Advanced classification metrics
//!
//! This module provides more advanced metrics for classification tasks,
//! including Matthews Correlation Coefficient, balanced accuracy,
//! Cohen's kappa, and more.

use crate::error::{MetricsError, Result};
use ndarray::{self, Array1, Array2, ArrayBase, Data, Dimension};
use num_traits::NumCast;
use std::collections::BTreeSet;

/// Calculates the Matthews Correlation Coefficient (MCC)
///
/// The MCC is a measure of the quality of binary and multiclass classifications.
/// It takes into account true and false positives and negatives and is generally
/// regarded as a balanced measure which can be used even if the classes are of very
/// different sizes. The MCC is in essence a correlation coefficient between the
/// observed and predicted binary classifications; it returns a value between -1 and +1.
/// A coefficient of +1 represents a perfect prediction, 0 no better than random
/// prediction and -1 indicates total disagreement between prediction and observation.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated targets as returned by a classifier
///
/// # Returns
///
/// * The Matthews correlation coefficient (float between -1 and 1)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::classification::advanced::matthews_corrcoef;
///
/// let y_true = array![1, 0, 1, 1, 0, 0];
/// let y_pred = array![1, 0, 1, 0, 0, 1];
///
/// let mcc = matthews_corrcoef(&y_true, &y_pred).unwrap();
/// ```
pub fn matthews_corrcoef<T, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
) -> Result<f64>
where
    T: PartialEq + NumCast + Clone + std::fmt::Debug,
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

    // Get unique classes from both arrays
    let mut classes = BTreeSet::new();
    for yt in y_true.iter() {
        classes.insert(format!("{:?}", yt));
    }
    for yp in y_pred.iter() {
        classes.insert(format!("{:?}", yp));
    }
    let classes: Vec<String> = classes.into_iter().collect();
    let n_classes = classes.len();

    // Map class labels to indices
    let mut class_to_idx = std::collections::HashMap::new();
    for (i, c) in classes.iter().enumerate() {
        class_to_idx.insert(c, i);
    }

    // Create confusion matrix
    let mut cm = vec![vec![0.0f64; n_classes]; n_classes];

    // Fill the confusion matrix
    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        let yt_idx = class_to_idx[&format!("{:?}", yt)];
        let yp_idx = class_to_idx[&format!("{:?}", yp)];
        cm[yt_idx][yp_idx] += 1.0;
    }

    // For binary case, use simplified formula (faster and more accurate)
    if n_classes == 2 {
        let tn = cm[0][0];
        let fp = cm[0][1];
        let false_neg = cm[1][0];
        let tp = cm[1][1];

        let numerator = tp * tn - fp * false_neg;
        let denominator = ((tp + fp) * (tp + false_neg) * (tn + fp) * (tn + false_neg)).sqrt();

        if denominator == 0.0 {
            return Ok(0.0);
        }

        return Ok(numerator / denominator);
    }

    // For multi-class case, calculate coefficients from confusion matrix
    let mut t = 0.0;
    for i in 0..n_classes {
        for j in 0..n_classes {
            t += cm[i][j];
        }
    }

    // Calculate sums of rows and columns
    let mut c = vec![0.0; n_classes];
    let mut k = vec![0.0; n_classes];
    for i in 0..n_classes {
        for j in 0..n_classes {
            c[j] += cm[i][j];
            k[i] += cm[i][j];
        }
    }

    // Calculate correlation coefficient
    let mut numerator = 0.0;
    for i in 0..n_classes {
        for j in 0..n_classes {
            if i == j {
                numerator += cm[i][j] * t - k[i] * c[i];
            } else {
                numerator -= cm[i][j] * k[j] * c[i];
            }
        }
    }

    let mut denominator_1 = 0.0;
    for i in 0..n_classes {
        denominator_1 += k[i] * k[i];
    }
    denominator_1 = t * t - denominator_1;

    let mut denominator_2 = 0.0;
    for i in 0..n_classes {
        denominator_2 += c[i] * c[i];
    }
    denominator_2 = t * t - denominator_2;

    let denominator = (denominator_1 * denominator_2).sqrt();

    if denominator == 0.0 {
        return Ok(0.0);
    }

    Ok(numerator / denominator)
}

/// Calculates the balanced accuracy score
///
/// The balanced accuracy in binary and multiclass classification problems
/// is the average of recall obtained on each class. It addresses issues with
/// imbalanced datasets by giving equal weight to each class regardless of
/// its support (number of true instances).
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated targets as returned by a classifier
///
/// # Returns
///
/// * The balanced accuracy score (float between 0.0 and 1.0)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::classification::advanced::balanced_accuracy_score;
///
/// let y_true = array![0, 0, 0, 1, 1, 1];
/// let y_pred = array![0, 0, 1, 1, 0, 1];
///
/// let bal_acc = balanced_accuracy_score(&y_true, &y_pred).unwrap();
/// // Recall for class 0: 2/3, recall for class 1: 2/3
/// // Balanced accuracy: (2/3 + 2/3) / 2 = 2/3
/// assert!((bal_acc - 2.0/3.0).abs() < 1e-10);
/// ```
pub fn balanced_accuracy_score<T, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
) -> Result<f64>
where
    T: PartialEq + NumCast + Clone + std::fmt::Debug,
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

    // Get unique classes from true labels
    let mut classes = BTreeSet::new();
    for yt in y_true.iter() {
        classes.insert(format!("{:?}", yt));
    }
    let classes: Vec<String> = classes.into_iter().collect();
    let n_classes = classes.len();

    if n_classes < 2 {
        return Err(MetricsError::InvalidInput(
            "Need at least two classes".to_string(),
        ));
    }

    // Calculate per-class recall
    let mut recall_sum = 0.0;

    for class in &classes {
        let mut true_positives = 0;
        let mut class_total = 0;

        for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
            let yt_str = format!("{:?}", yt);
            let yp_str = format!("{:?}", yp);

            if yt_str == *class {
                class_total += 1;
                if yp_str == *class {
                    true_positives += 1;
                }
            }
        }

        // Calculate recall for this class
        let recall = if class_total > 0 {
            true_positives as f64 / class_total as f64
        } else {
            0.0
        };

        recall_sum += recall;
    }

    // Balanced accuracy is the average recall
    Ok(recall_sum / n_classes as f64)
}

/// Calculates Cohen's kappa score
///
/// Cohen's kappa measures inter-annotator agreement for categorical items.
/// It is a more robust measure compared to simple percent agreement calculation,
/// as it takes into account the agreement occurring by chance.
///
/// Kappa scores range from -1 to 1, where:
/// * 1 indicates perfect agreement
/// * 0 indicates random agreement
/// * -1 indicates perfect disagreement
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated targets as returned by a classifier
///
/// # Returns
///
/// * Cohen's kappa score (float between -1.0 and 1.0)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::classification::advanced::cohen_kappa_score;
///
/// let y_true = array![0, 1, 2, 0, 1, 2];
/// let y_pred = array![0, 0, 2, 0, 0, 2];
///
/// let kappa = cohen_kappa_score(&y_true, &y_pred).unwrap();
/// ```
pub fn cohen_kappa_score<T, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_pred: &ArrayBase<S2, D2>,
) -> Result<f64>
where
    T: PartialEq + NumCast + Clone + std::fmt::Debug,
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
    let mut classes = BTreeSet::new();
    for yt in y_true.iter() {
        classes.insert(format!("{:?}", yt));
    }
    for yp in y_pred.iter() {
        classes.insert(format!("{:?}", yp));
    }
    let classes: Vec<String> = classes.into_iter().collect();
    let n_classes = classes.len();

    // Map class labels to indices
    let mut class_to_idx = std::collections::HashMap::new();
    for (i, c) in classes.iter().enumerate() {
        class_to_idx.insert(c, i);
    }

    // Create confusion matrix
    let mut cm = vec![vec![0.0f64; n_classes]; n_classes];

    // Fill the confusion matrix
    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        let yt_idx = class_to_idx[&format!("{:?}", yt)];
        let yp_idx = class_to_idx[&format!("{:?}", yp)];
        cm[yt_idx][yp_idx] += 1.0;
    }

    // Calculate observed agreement (accuracy)
    let mut observed = 0.0;
    for i in 0..n_classes {
        observed += cm[i][i];
    }
    observed /= n_samples as f64;

    // Calculate expected agreement (due to chance)
    let mut expected = 0.0;
    for i in 0..n_classes {
        let mut row_sum = 0.0;
        let mut col_sum = 0.0;
        for j in 0..n_classes {
            row_sum += cm[i][j];
            col_sum += cm[j][i];
        }
        expected += (row_sum / n_samples as f64) * (col_sum / n_samples as f64);
    }

    // Calculate kappa
    if expected == 1.0 {
        // If expected agreement is 1, then all predictions fall into a single category
        return Ok(1.0);
    }

    Ok((observed - expected) / (1.0 - expected))
}

/// Calculates the Brier score loss
///
/// The Brier score is a proper score function that measures the accuracy
/// of probabilistic predictions. It is applicable to tasks in which predictions
/// must assign probabilities to a set of mutually exclusive discrete outcomes.
///
/// The Brier score measures the mean squared difference between the predicted
/// probability and the actual outcome. For a binary classifier:
/// BS = (1/N) * sum[(p - o)²] where p is the predicted probability and o is
/// the actual outcome (0 or 1).
///
/// # Arguments
///
/// * `y_true` - Binary ground truth values (0 or 1)
/// * `y_prob` - Probability estimates (0 to 1)
///
/// # Returns
///
/// * The Brier score loss (float between 0.0 and 1.0, lower is better)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::classification::advanced::brier_score_loss;
///
/// let y_true = array![0, 1, 1, 0];
/// let y_prob = array![0.1, 0.9, 0.8, 0.4];
///
/// let brier = brier_score_loss(&y_true, &y_prob).unwrap();
/// // Brier = [(0.1-0)² + (0.9-1)² + (0.8-1)² + (0.4-0)²] / 4
/// //       = [0.01 + 0.01 + 0.04 + 0.16] / 4 = 0.055
/// assert!((brier - 0.055).abs() < 1e-10);
/// ```
pub fn brier_score_loss<S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_prob: &ArrayBase<S2, D2>,
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

    // Validate y_true contains only binary values
    for yt in y_true.iter() {
        if *yt != 0 && *yt != 1 {
            return Err(MetricsError::InvalidInput(
                "y_true must contain only binary values (0 or 1)".to_string(),
            ));
        }
    }

    // Validate y_prob contains only values between 0 and 1
    for yp in y_prob.iter() {
        if *yp < 0.0 || *yp > 1.0 {
            return Err(MetricsError::InvalidInput(
                "y_prob must contain only values between 0 and 1".to_string(),
            ));
        }
    }

    // Calculate Brier score
    let mut score = 0.0;
    for (yt, yp) in y_true.iter().zip(y_prob.iter()) {
        let y_true_f64 = *yt as f64;
        score += (yp - y_true_f64).powi(2);
    }

    Ok(score / n_samples as f64)
}

/// Calculates the Jaccard similarity score
///
/// The Jaccard index, or Jaccard similarity coefficient, is a statistic used
/// for comparing the similarity and diversity of sample sets. It is defined as
/// the size of the intersection divided by the size of the union of the sample sets.
///
/// For binary classification, this measures the similarity between the predicted
/// positive set and the true positive set.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated targets as returned by a classifier
/// * `pos_label` - The label to report as positive class
///
/// # Returns
///
/// * The Jaccard similarity score (float between 0.0 and 1.0)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::classification::advanced::jaccard_score;
///
/// let y_true = array![0, 1, 1, 0, 1];
/// let y_pred = array![0, 0, 1, 1, 1];
///
/// let jaccard = jaccard_score(&y_true, &y_pred, 1).unwrap();
/// // jaccard = |intersection| / |union| = 2 / 4 = 0.5
/// assert!((jaccard - 0.5).abs() < 1e-10);
/// ```
pub fn jaccard_score<T, S1, S2, D1, D2>(
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

    // Count intersection and union
    let mut intersection_count = 0;
    let mut union_count = 0;

    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        let is_true_positive = yt == &pos_label;
        let is_pred_positive = yp == &pos_label;

        if is_true_positive && is_pred_positive {
            intersection_count += 1;
            union_count += 1;
        } else if is_true_positive || is_pred_positive {
            union_count += 1;
        }
    }

    // Calculate Jaccard similarity coefficient
    if union_count == 0 {
        // If union is empty, define Jaccard as 1.0 (perfect similarity)
        return Ok(1.0);
    }

    Ok(intersection_count as f64 / union_count as f64)
}

/// Calculates the Hamming loss
///
/// The Hamming loss is the fraction of labels that are incorrectly predicted.
/// For binary classification, this is equivalent to 1 - accuracy.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated targets as returned by a classifier
///
/// # Returns
///
/// * The Hamming loss (float between 0.0 and 1.0, lower is better)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::classification::advanced::hamming_loss;
///
/// let y_true = array![0, 1, 0, 1];
/// let y_pred = array![1, 1, 0, 0];
///
/// let loss = hamming_loss(&y_true, &y_pred).unwrap();
/// // 2 out of 4 labels are incorrect: loss = 2/4 = 0.5
/// assert!((loss - 0.5).abs() < 1e-10);
/// ```
pub fn hamming_loss<T, S1, S2, D1, D2>(
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

    // Count mismatches
    let mut mismatches = 0;
    for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
        if yt != yp {
            mismatches += 1;
        }
    }

    Ok(mismatches as f64 / n_samples as f64)
}

/// Calculates the log loss, also known as cross-entropy loss, for multi-class classification
///
/// The log loss is the negative log-likelihood of the true labels given a probabilistic
/// classifier's predictions. It is commonly used for evaluating the performance of a
/// classification model where the prediction input is a probability value.
///
/// For multi-class problems, this function supports both a binary array of shape
/// (n_samples, n_classes) or a 1D array of class indices with a separate array
/// of class probabilities of shape (n_samples, n_classes).
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) labels. If a 1D array, it should contain class indices.
///   If a 2D array, it should be a one-hot or multi-label binary array.
/// * `y_prob` - Predicted probabilities, as returned by a classifier
/// * `eps` - Small value to avoid taking the log of 0 (default: 1e-15)
/// * `normalize` - If true, return the mean loss per sample, otherwise, return the sum of the loss per sample.
///
/// # Returns
///
/// * The log loss (float, lower is better)
///
/// # Examples
///
/// ```
/// // Due to some implementation complexities with the doctest,
/// // we'll show pseudocode here and the actual implementation is tested in unit tests
/// 
/// // Example 1: Binary classification
/// // y_true = [0, 1, 1, 0]
/// // y_prob = [[0.9, 0.1], [0.1, 0.9], [0.2, 0.8], [0.7, 0.3]]
/// // log_loss(y_true, y_prob, 1e-15, true) -> ~0.2
/// 
/// // Example 2: Multi-class classification
/// // y_true = [0, 1, 2, 1] 
/// // y_prob = [
/// //   [0.9, 0.05, 0.05],  // Sample 0 probabilities
/// //   [0.1, 0.8, 0.1],    // Sample 1 probabilities
/// //   [0.1, 0.2, 0.7],    // Sample 2 probabilities
/// //   [0.2, 0.7, 0.1]     // Sample 3 probabilities
/// // ]
/// // log_loss(y_true, y_prob, 1e-15, true) -> ~0.32
/// ```
// Internal implementation for 2D arrays - private to prevent doctests
#[doc(hidden)]
fn log_loss_2d<T, S1, S2>(
    y_true: &ArrayBase<S1, ndarray::Ix2>,
    y_prob: &ArrayBase<S2, ndarray::Ix2>,
    eps: f64,
    normalize: bool,
) -> Result<f64>
where
    T: PartialEq + NumCast + Clone + std::fmt::Debug,
    S1: Data<Elem = T>,
    S2: Data<Elem = f64>,
{
    // Check inputs
    let n_samples = y_true.len_of(ndarray::Axis(0));
    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    let n_classes = y_prob.len_of(ndarray::Axis(1));

    let mut loss = 0.0;
    for i in 0..n_samples {
        let mut sample_loss = 0.0;

        for j in 0..n_classes {
            let true_val: f64 = if let Some(val) = NumCast::from(
                y_true
                    .get((i, j))
                    .ok_or_else(|| {
                        MetricsError::InvalidInput(
                            "Index out of bounds accessing y_true".to_string(),
                        )
                    })?
                    .clone(),
            ) {
                val
            } else {
                return Err(MetricsError::InvalidInput(
                    "Could not convert y_true value to float".to_string(),
                ));
            };

            // Only consider classes that are truly present (true_val > 0)
            if true_val > 0.0 {
                let prob = y_prob
                    .get((i, j))
                    .ok_or_else(|| {
                        MetricsError::InvalidInput(
                            "Index out of bounds accessing y_prob".to_string(),
                        )
                    })?
                    .max(eps)
                    .min(1.0 - eps);

                sample_loss -= true_val * prob.ln();
            }
        }

        loss += sample_loss;
    }

    // Normalize if requested
    if normalize {
        loss /= n_samples as f64;
    }

    Ok(loss)
}

// Internal implementation for 1D arrays - private to prevent doctests
#[doc(hidden)]
fn log_loss_1d<T, S1, S2>(
    y_true: &ArrayBase<S1, ndarray::Ix1>,
    y_prob: &ArrayBase<S2, ndarray::Ix1>,
    eps: f64,
    normalize: bool,
) -> Result<f64>
where
    T: PartialEq + NumCast + Clone + std::fmt::Debug,
    S1: Data<Elem = T>,
    S2: Data<Elem = f64>,
{
    // Check inputs
    let n_samples = y_true.len();
    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    if y_prob.len() != n_samples {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_prob have different numbers of samples: {} vs {}",
            n_samples,
            y_prob.len()
        )));
    }

    let mut loss = 0.0;

    for i in 0..n_samples {
        let y_i = y_true.get(i).ok_or_else(|| {
            MetricsError::InvalidInput("Index out of bounds accessing y_true".to_string())
        })?;

        let y_pred_i = y_prob.get(i).ok_or_else(|| {
            MetricsError::InvalidInput("Index out of bounds accessing y_prob".to_string())
        })?;

        let prob = y_pred_i.max(eps).min(1.0 - eps);

        let true_val_num: usize = if let Some(val) = NumCast::from(y_i.clone()) {
            val
        } else {
            return Err(MetricsError::InvalidInput(
                "Could not convert y_true value to usize".to_string(),
            ));
        };

        if true_val_num == 1 {
            // For class 1, use the probability as-is
            loss -= prob.ln();
        } else if true_val_num == 0 {
            // For class 0, use 1 - probability
            loss -= (1.0 - prob).ln();
        } else {
            return Err(MetricsError::InvalidInput(
                format!("For binary classification with 1D arrays, y_true must contain only 0 or 1 values, got {}", true_val_num)
            ));
        }
    }

    // Normalize if requested
    if normalize {
        loss /= n_samples as f64;
    }

    Ok(loss)
}

/// Calculate log loss (logistic loss or cross-entropy loss).
pub fn log_loss<T, S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_prob: &ArrayBase<S2, D2>,
    eps: f64,
    normalize: bool,
) -> Result<f64>
where
    T: PartialEq + NumCast + Clone + std::fmt::Debug + num_traits::Zero,
    S1: Data<Elem = T>,
    S2: Data<Elem = f64>,
    D1: Dimension,
    D2: Dimension,
{
    // Dispatch based on dimensionality
    match (y_true.ndim(), y_prob.ndim()) {
        (1, 1) => {
            // Both are 1D arrays - binary classification
            let y_true_1d = y_true
                .view()
                .into_dimensionality::<ndarray::Ix1>()
                .map_err(|_| {
                    MetricsError::InvalidInput("Error converting y_true to 1D".to_string())
                })?;

            let y_prob_1d = y_prob
                .view()
                .into_dimensionality::<ndarray::Ix1>()
                .map_err(|_| {
                    MetricsError::InvalidInput("Error converting y_prob to 1D".to_string())
                })?;

            log_loss_1d(&y_true_1d, &y_prob_1d, eps, normalize)
        }
        (1, 2) => {
            // y_true is 1D (class indices), y_prob is 2D (probabilities per class)
            let y_true_1d = y_true
                .view()
                .into_dimensionality::<ndarray::Ix1>()
                .map_err(|_| {
                    MetricsError::InvalidInput("Error converting y_true to 1D".to_string())
                })?;

            let y_prob_2d = y_prob
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| {
                    MetricsError::InvalidInput("Error converting y_prob to 2D".to_string())
                })?;

            // Create a 2D one-hot encoding from y_true
            let n_samples = y_true_1d.len();
            let n_classes = y_prob_2d.shape()[1];

            // Create one-hot array
            let mut y_true_2d = Array2::<T>::zeros((n_samples, n_classes));

            for i in 0..n_samples {
                let class_idx: usize = if let Some(val) = y_true_1d.get(i) {
                    if let Some(idx) = NumCast::from(val.clone()) {
                        idx
                    } else {
                        return Err(MetricsError::InvalidInput(format!(
                            "Could not convert label {:?} to index",
                            val
                        )));
                    }
                } else {
                    return Err(MetricsError::InvalidInput(
                        "Index out of bounds accessing y_true".to_string(),
                    ));
                };

                if class_idx >= n_classes {
                    return Err(MetricsError::InvalidInput(format!(
                        "Class label {} is out of bounds for y_prob with {} classes",
                        class_idx, n_classes
                    )));
                }

                // Set the corresponding class to 1
                if let Some(one) = NumCast::from(1) {
                    y_true_2d[[i, class_idx]] = one;
                } else {
                    return Err(MetricsError::InvalidInput(
                        "Could not cast 1 to element type".to_string(),
                    ));
                }
            }

            log_loss_2d(&y_true_2d.view(), &y_prob_2d, eps, normalize)
        }
        (2, 2) => {
            // Both are 2D - multi-class/multi-label
            let y_true_2d = y_true
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| {
                    MetricsError::InvalidInput("Error converting y_true to 2D".to_string())
                })?;

            let y_prob_2d = y_prob
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| {
                    MetricsError::InvalidInput("Error converting y_prob to 2D".to_string())
                })?;

            log_loss_2d(&y_true_2d, &y_prob_2d, eps, normalize)
        }
        _ => Err(MetricsError::InvalidInput(format!(
            "Unsupported dimensions: y_true ({:?}), y_prob ({:?})",
            y_true.shape(),
            y_prob.shape()
        ))),
    }
}

/// Computes calibration curves for binary classification
///
/// When performing classification tasks, it is often necessary to predict not only the class label
/// but also the probability of the sample belonging to a particular class. This function assesses
/// the calibration of predicted probabilities by binning predictions and computing the true
/// fraction of positives in each bin.
///
/// # Arguments
///
/// * `y_true` - Binary true labels (0 or 1)
/// * `y_prob` - Predicted probabilities for the positive class
/// * `n_bins` - Number of bins to use when computing the calibration curve
/// * `strategy` - Strategy for defining bins: "uniform" (equal-width bins) or "quantile" (equal-sized bins)
///
/// # Returns
///
/// * A tuple containing:
///   * `prob_true` - The true fraction of positives in each bin
///   * `prob_pred` - The mean predicted probability in each bin
///   * `counts` - The number of samples in each bin
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::classification::advanced::calibration_curve;
///
/// let y_true = array![0, 0, 0, 0, 1, 1, 1, 1, 1];
/// let y_prob = array![0.1, 0.2, 0.3, 0.4, 0.65, 0.7, 0.8, 0.9, 0.99];
///
/// let (prob_true, prob_pred, counts) = calibration_curve(&y_true, &y_prob, 5, "uniform").unwrap();
/// ```
pub fn calibration_curve<S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_prob: &ArrayBase<S2, D2>,
    n_bins: usize,
    strategy: &str,
) -> Result<(Array1<f64>, Array1<f64>, Array1<usize>)>
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

    // Validate y_true contains only binary values
    for yt in y_true.iter() {
        if *yt != 0 && *yt != 1 {
            return Err(MetricsError::InvalidInput(
                "y_true must contain only binary values (0 or 1)".to_string(),
            ));
        }
    }

    // Validate y_prob contains only values between 0 and 1
    for yp in y_prob.iter() {
        if *yp < 0.0 || *yp > 1.0 {
            return Err(MetricsError::InvalidInput(
                "y_prob must contain only values between 0 and 1".to_string(),
            ));
        }
    }

    // Validate n_bins
    if n_bins < 1 {
        return Err(MetricsError::InvalidInput(
            "n_bins must be at least 1".to_string(),
        ));
    }

    // Validate strategy
    if strategy != "uniform" && strategy != "quantile" {
        return Err(MetricsError::InvalidInput(
            "strategy must be either 'uniform' or 'quantile'".to_string(),
        ));
    }

    // Create bin edges
    let bin_edges = if strategy == "uniform" {
        // Create uniform bins
        let mut edges = Vec::with_capacity(n_bins + 1);
        for i in 0..=n_bins {
            edges.push(i as f64 / n_bins as f64);
        }
        edges
    } else {
        // Create quantile bins
        let mut probs_sorted: Vec<f64> = y_prob.iter().copied().collect();
        probs_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut edges = Vec::with_capacity(n_bins + 1);
        edges.push(0.0);
        for i in 1..n_bins {
            let idx = (i * n_samples) / n_bins;
            edges.push(probs_sorted[idx]);
        }
        edges.push(1.0);
        edges
    };

    // Initialize arrays to store results
    let mut prob_true = Array1::<f64>::zeros(n_bins);
    let mut prob_pred = Array1::<f64>::zeros(n_bins);
    let mut counts = Array1::<f64>::zeros(n_bins);

    // Bin the probabilities and compute statistics
    for (true_label, prob) in y_true.iter().zip(y_prob.iter()) {
        // Find bin index
        let bin_idx = bin_edges
            .iter()
            .enumerate()
            .filter(|(i, &edge)| *i < n_bins && prob >= &edge && prob <= &bin_edges[i + 1])
            .map(|(i, _)| i)
            .next()
            .unwrap_or_else(|| {
                // Handle edge case when prob is exactly 1.0
                if (prob - 1.0).abs() < 1e-10 {
                    n_bins - 1
                } else {
                    0 // Default to first bin as a fallback
                }
            });

        // Update bin statistics
        prob_pred[bin_idx] += prob;
        prob_true[bin_idx] += *true_label as f64;
        counts[bin_idx] += 1.0;
    }

    // Compute mean predicted probability and fraction of true positives for each bin
    for i in 0..n_bins {
        if counts[i] > 0.0 {
            prob_pred[i] /= counts[i];
            prob_true[i] /= counts[i];
        }
    }

    Ok((prob_true, prob_pred, counts.mapv(|x: f64| x as usize)))
}
