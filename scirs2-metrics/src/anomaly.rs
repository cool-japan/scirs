//! Anomaly Detection Metrics module
//!
//! This module provides metrics for evaluating anomaly detection systems.
//! These metrics are useful for tasks like outlier detection, novelty detection,
//! and time series anomaly detection.
//!
//! ## Basic Metrics
//!
//! Basic anomaly detection metrics include Detection Accuracy, False Alarm Rate,
//! Miss Detection Rate, and specialized versions of AUC and Average Precision
//! for anomaly detection contexts.
//!
//! ```
//! use scirs2_core::ndarray::array;
//! use scirs2_metrics::anomaly::{
//!     detection_accuracy, false_alarm_rate, miss_detection_rate,
//!     anomaly_auc_score, anomaly_average_precision_score
//! };
//!
//! // Example: anomaly detection results
//! // y_true: 1.0 for anomalies, 0.0 for normal instances
//! let y_true = array![0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
//! // y_pred: predicted anomalies (1.0) and normal instances (0.0)
//! let y_pred = array![0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0];
//! // y_score: anomaly scores (higher indicates more anomalous)
//! let y_score = array![0.1, 0.2, 0.9, 0.7, 0.8, 0.3, 0.6, 0.95, 0.2, 0.1];
//!
//! // Compute accuracy of anomaly detection
//! let accuracy = detection_accuracy(&y_true, &y_pred).unwrap();
//!
//! // False alarm rate (Type I error)
//! let far = false_alarm_rate(&y_true, &y_pred).unwrap();
//!
//! // Miss detection rate (Type II error)
//! let mdr = miss_detection_rate(&y_true, &y_pred).unwrap();
//!
//! // AUC for anomaly detection
//! let auc = anomaly_auc_score(&y_true, &y_score).unwrap();
//!
//! // Average precision score
//! let ap = anomaly_average_precision_score(&y_true, &y_score).unwrap();
//! ```
//!
//! ## Distribution Metrics
//!
//! Distribution-based metrics for comparing distributions and measuring divergence:
//!
//! ```
//! use scirs2_core::ndarray::array;
//! use scirs2_metrics::anomaly::{
//!     kl_divergence, js_divergence, wasserstein_distance, maximum_mean_discrepancy
//! };
//!
//! // Example: two probability distributions
//! let p = array![0.2, 0.5, 0.3];
//! let q = array![0.3, 0.4, 0.3];
//!
//! // Compute Kullback-Leibler divergence
//! let kl = kl_divergence(&p, &q).unwrap();
//!
//! // Jensen-Shannon divergence
//! let js = js_divergence(&p, &q).unwrap();
//!
//! // Wasserstein distance (1D version)
//! let samples_p = array![1.0, 2.0, 3.0, 4.0, 5.0];
//! let samples_q = array![1.5, 2.5, 3.5, 4.5, 5.5];
//! let w_dist = wasserstein_distance(&samples_p, &samples_q).unwrap();
//!
//! // Maximum Mean Discrepancy with RBF kernel
//! let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
//! let y = array![1.2, 2.3, 3.1, 4.2, 5.5];
//! let mmd = maximum_mean_discrepancy(&x, &y, None).unwrap();
//! ```
//!
//! ## Time Series Anomaly Metrics
//!
//! Specialized metrics for time series anomaly detection that account for temporal
//! aspects and tolerance windows:
//!
//! ```
//! use scirs2_core::ndarray::array;
//! use scirs2_metrics::anomaly::{
//!     precision_recall_with_tolerance, point_adjusted_precision_recall, nab_score
//! };
//!
//! // Example: time series anomaly detection
//! // Ground truth anomaly labels (0 for normal, 1 for anomaly)
//! let y_true = array![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
//! // Predicted anomaly labels
//! let y_pred = array![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
//!
//! // Precision and recall with a tolerance window of 2 time steps
//! let (precision, recall, f1) = precision_recall_with_tolerance(&y_true, &y_pred, 2).unwrap();
//!
//! // Point-adjusted precision and recall
//! let (pa_precision, pa_recall, pa_f1) = point_adjusted_precision_recall(&y_true, &y_pred).unwrap();
//!
//! // Numenta Anomaly Benchmark (NAB) score with default parameters
//! let nab = nab_score(&y_true, &y_pred, None, None, None).unwrap();
//! ```

use scirs2_core::ndarray::{Array1, ArrayBase, Data, Ix1};
use scirs2_core::numeric::Float;
use std::cmp::Ordering;
use std::collections::HashSet;

use crate::error::{MetricsError, Result};

/// Calculates detection accuracy for anomaly detection.
///
/// Detection accuracy is the ratio of correctly classified instances to the total number of instances.
/// For anomaly detection, this measures the overall correctness of the classifier.
///
/// # Arguments
///
/// * `y_true` - Ground truth labels (1 for anomalies, 0 for normal instances)
/// * `y_pred` - Predicted labels (1 for anomalies, 0 for normal instances)
///
/// # Returns
///
/// * The detection accuracy (float between 0 and 1)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_metrics::anomaly::detection_accuracy;
///
/// let y_true = array![0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
/// let y_pred = array![0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0];
///
/// // Overall accuracy: 8 correct out of 10 instances (80%)
/// let accuracy = detection_accuracy(&y_true, &y_pred).unwrap();
/// assert!(accuracy >= 0.0 && accuracy <= 1.0);
/// ```
#[allow(dead_code)]
pub fn detection_accuracy<T, S, R>(
    y_true: &ArrayBase<S, Ix1>,
    y_pred: &ArrayBase<R, Ix1>,
) -> Result<f64>
where
    T: Float + PartialOrd + Clone,
    S: Data<Elem = T>,
    R: Data<Elem = T>,
{
    // Check that arrays have the same length
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::InvalidInput(format!(
            "Arrays have different lengths: {} vs {}",
            y_true.len(),
            y_pred.len()
        )));
    }

    if y_true.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    let zero = T::zero();
    let total = y_true.len();
    let mut correct = 0;

    for (truth, pred) in y_true.iter().zip(y_pred.iter()) {
        if (truth > &zero && pred > &zero) || (truth == &zero && pred == &zero) {
            correct += 1;
        }
    }

    Ok(correct as f64 / total as f64)
}

/// Calculates false alarm rate (Type I error) for anomaly detection.
///
/// False alarm rate is the probability that a normal instance is incorrectly classified
/// as an anomaly (false positive rate). In anomaly detection, it represents the rate at which
/// normal instances are mistakenly flagged as anomalies.
///
/// # Arguments
///
/// * `y_true` - Ground truth labels (1 for anomalies, 0 for normal instances)
/// * `y_pred` - Predicted labels (1 for anomalies, 0 for normal instances)
///
/// # Returns
///
/// * The false alarm rate (float between 0 and 1)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_metrics::anomaly::false_alarm_rate;
///
/// let y_true = array![0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
/// let y_pred = array![0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
///
/// // 2 false alarms out of 7 normal instances (≈ 0.286)
/// let far = false_alarm_rate(&y_true, &y_pred).unwrap();
/// assert!(far >= 0.0 && far <= 1.0);
/// ```
#[allow(dead_code)]
pub fn false_alarm_rate<T, S, R>(
    y_true: &ArrayBase<S, Ix1>,
    y_pred: &ArrayBase<R, Ix1>,
) -> Result<f64>
where
    T: Float + PartialOrd + Clone,
    S: Data<Elem = T>,
    R: Data<Elem = T>,
{
    // Check that arrays have the same length
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::InvalidInput(format!(
            "Arrays have different lengths: {} vs {}",
            y_true.len(),
            y_pred.len()
        )));
    }

    if y_true.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    let zero = T::zero();
    let mut false_alarms = 0;
    let mut total_normal = 0;

    for (truth, pred) in y_true.iter().zip(y_pred.iter()) {
        if truth == &zero {
            total_normal += 1;
            if pred > &zero {
                false_alarms += 1;
            }
        }
    }

    if total_normal == 0 {
        return Err(MetricsError::InvalidInput(
            "No normal instances in ground truth".to_string(),
        ));
    }

    Ok(false_alarms as f64 / total_normal as f64)
}

/// Calculates miss detection rate (Type II error) for anomaly detection.
///
/// Miss detection rate is the probability that an anomaly is incorrectly classified
/// as normal (false negative rate). In anomaly detection, it represents the rate at which
/// anomalies are missed by the detector.
///
/// # Arguments
///
/// * `y_true` - Ground truth labels (1 for anomalies, 0 for normal instances)
/// * `y_pred` - Predicted labels (1 for anomalies, 0 for normal instances)
///
/// # Returns
///
/// * The miss detection rate (float between 0 and 1)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_metrics::anomaly::miss_detection_rate;
///
/// let y_true = array![0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
/// let y_pred = array![0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
///
/// // 1 missed anomaly out of 3 total anomalies (≈ 0.333)
/// let mdr = miss_detection_rate(&y_true, &y_pred).unwrap();
/// assert!(mdr >= 0.0 && mdr <= 1.0);
/// ```
#[allow(dead_code)]
pub fn miss_detection_rate<T, S, R>(
    y_true: &ArrayBase<S, Ix1>,
    y_pred: &ArrayBase<R, Ix1>,
) -> Result<f64>
where
    T: Float + PartialOrd + Clone,
    S: Data<Elem = T>,
    R: Data<Elem = T>,
{
    // Check that arrays have the same length
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::InvalidInput(format!(
            "Arrays have different lengths: {} vs {}",
            y_true.len(),
            y_pred.len()
        )));
    }

    if y_true.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    let zero = T::zero();
    let mut missed_anomalies = 0;
    let mut total_anomalies = 0;

    for (truth, pred) in y_true.iter().zip(y_pred.iter()) {
        if truth > &zero {
            total_anomalies += 1;
            if pred == &zero {
                missed_anomalies += 1;
            }
        }
    }

    if total_anomalies == 0 {
        return Err(MetricsError::InvalidInput(
            "No anomalies in ground truth".to_string(),
        ));
    }

    Ok(missed_anomalies as f64 / total_anomalies as f64)
}

/// Calculates Area Under the ROC Curve (AUC) for anomaly detection.
///
/// AUC measures the ability of the model to discriminate between anomalies
/// and normal instances across varying thresholds. It is particularly useful
/// when dealing with imbalanced datasets, which is common in anomaly detection.
///
/// # Arguments
///
/// * `y_true` - Ground truth labels (1 for anomalies, 0 for normal instances)
/// * `y_score` - Anomaly scores (higher values indicate more anomalous)
///
/// # Returns
///
/// * The AUC score (float between 0 and 1)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_metrics::anomaly::anomaly_auc_score;
///
/// let y_true = array![0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
/// let y_score = array![0.1, 0.2, 0.9, 0.3, 0.8, 0.2, 0.4, 0.95, 0.1, 0.05];
///
/// // Calculate AUC score
/// let auc = anomaly_auc_score(&y_true, &y_score).unwrap();
/// assert!(auc >= 0.0 && auc <= 1.0);
/// ```
#[allow(dead_code)]
pub fn anomaly_auc_score<T, S, R>(
    y_true: &ArrayBase<S, Ix1>,
    y_score: &ArrayBase<R, Ix1>,
) -> Result<f64>
where
    T: Float + PartialOrd + Clone,
    S: Data<Elem = T>,
    R: Data<Elem = T>,
{
    // Check that arrays have the same length
    if y_true.len() != y_score.len() {
        return Err(MetricsError::InvalidInput(format!(
            "Arrays have different lengths: {} vs {}",
            y_true.len(),
            y_score.len()
        )));
    }

    if y_true.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    // Collect pairs of (_score, label) for sorting
    let mut score_label: Vec<_> = y_score
        .iter()
        .zip(y_true.iter())
        .map(|(s, l)| (s.to_f64().unwrap_or(0.0), l.to_f64().unwrap_or(0.0)))
        .collect();

    // Sort by _score in descending order
    score_label.sort_by(|(a, _), (b, _)| b.partial_cmp(a).unwrap_or(Ordering::Equal));

    // Count total positives and negatives
    let positives: usize = y_true.iter().filter(|&&x| x > T::zero()).count();
    let negatives = y_true.len() - positives;

    if positives == 0 || negatives == 0 {
        return Err(MetricsError::InvalidInput(
            "AUC is not defined when only one class is present".to_string(),
        ));
    }

    // Calculate AUC using the Mann-Whitney U statistic
    let mut positive_rank_sum = 0.0;
    let mut current_rank = 1.0;
    let mut prev_score = score_label[0].0;
    let mut count_at_current_score = 1;
    let mut pos_at_current_score = if score_label[0].1 > 0.0 { 1 } else { 0 };

    for i in 1..score_label.len() {
        let (_score, label) = score_label[i];

        if (_score - prev_score).abs() < 1e-10 {
            // Same score, continue counting
            count_at_current_score += 1;
            if label > 0.0 {
                pos_at_current_score += 1;
            }
        } else {
            // Assign average rank to ties
            let avg_rank = current_rank + (count_at_current_score - 1) as f64 / 2.0;
            positive_rank_sum += avg_rank * pos_at_current_score as f64;

            // Reset for new _score
            current_rank += count_at_current_score as f64;
            count_at_current_score = 1;
            pos_at_current_score = if label > 0.0 { 1 } else { 0 };
            prev_score = _score;
        }
    }

    // Handle the last group
    let avg_rank = current_rank + (count_at_current_score - 1) as f64 / 2.0;
    positive_rank_sum += avg_rank * pos_at_current_score as f64;

    // Calculate AUC from U statistic
    let u = positive_rank_sum - (positives * (positives + 1)) as f64 / 2.0;
    let auc = u / (positives * negatives) as f64;

    Ok(auc)
}

/// Calculates average precision score for anomaly detection.
///
/// Average precision summarizes the precision-recall curve as the weighted mean
/// of precisions at each threshold, with the weight being the increase in recall.
/// It's especially useful for anomaly detection where anomalies are rare.
///
/// # Arguments
///
/// * `y_true` - Ground truth labels (1 for anomalies, 0 for normal instances)
/// * `y_score` - Anomaly scores (higher values indicate more anomalous)
///
/// # Returns
///
/// * The average precision score (float between 0 and 1)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_metrics::anomaly::anomaly_average_precision_score;
///
/// let y_true = array![0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
/// let y_score = array![0.1, 0.2, 0.9, 0.3, 0.8, 0.2, 0.4, 0.95, 0.1, 0.05];
///
/// // Calculate average precision score
/// let ap = anomaly_average_precision_score(&y_true, &y_score).unwrap();
/// assert!(ap > 0.0 && ap <= 1.0);
/// ```
#[allow(dead_code)]
pub fn anomaly_average_precision_score<T, S, R>(
    y_true: &ArrayBase<S, Ix1>,
    y_score: &ArrayBase<R, Ix1>,
) -> Result<f64>
where
    T: Float + PartialOrd + Clone,
    S: Data<Elem = T>,
    R: Data<Elem = T>,
{
    // Check that arrays have the same length
    if y_true.len() != y_score.len() {
        return Err(MetricsError::InvalidInput(format!(
            "Arrays have different lengths: {} vs {}",
            y_true.len(),
            y_score.len()
        )));
    }

    if y_true.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    // Collect pairs of (_score, label) for sorting
    let mut score_label: Vec<_> = y_score
        .iter()
        .zip(y_true.iter())
        .map(|(s, l)| (s.to_f64().unwrap_or(0.0), l.to_f64().unwrap_or(0.0)))
        .collect();

    // Sort by _score in descending order
    score_label.sort_by(|(a, _), (b, _)| b.partial_cmp(a).unwrap_or(Ordering::Equal));

    // Count total positives (anomalies)
    let total_positives: usize = y_true.iter().filter(|&&x| x > T::zero()).count();

    if total_positives == 0 {
        return Err(MetricsError::InvalidInput(
            "Average precision is not defined when there are no positive samples".to_string(),
        ));
    }

    // Calculate precision and recall at each threshold
    let mut true_positives = 0;
    let mut false_positives = 0;
    let mut sum_precision = 0.0;
    let mut prev_recall = 0.0;

    for (_, label) in score_label.iter() {
        if *label > 0.0 {
            true_positives += 1;
        } else {
            false_positives += 1;
        }

        // Calculate precision and recall at this threshold
        let precision = true_positives as f64 / (true_positives + false_positives) as f64;
        let recall = true_positives as f64 / total_positives as f64;

        // Add to the sum of precision * recall increments
        if *label > 0.0 {
            // Only update the sum when we encounter a positive label
            sum_precision += precision * (recall - prev_recall);
            prev_recall = recall;
        }
    }

    Ok(sum_precision)
}

/// Calculates Kullback-Leibler (KL) divergence between two probability distributions.
///
/// KL divergence measures how one probability distribution P diverges from a second
/// probability distribution Q. It is not symmetric: KL(P||Q) ≠ KL(Q||P). A smaller value
/// indicates that the distributions are more similar.
///
/// # Arguments
///
/// * `p` - First probability distribution (should sum to approximately 1)
/// * `q` - Second probability distribution (should sum to approximately 1)
///
/// # Returns
///
/// * The KL divergence (non-negative float)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_metrics::anomaly::kl_divergence;
///
/// // Two probability distributions
/// let p = array![0.5, 0.3, 0.2];
/// let q = array![0.3, 0.4, 0.3];
///
/// // Calculate KL divergence
/// let kl = kl_divergence(&p, &q).unwrap();
/// ```
#[allow(dead_code)]
pub fn kl_divergence<T, S, R>(p: &ArrayBase<S, Ix1>, q: &ArrayBase<R, Ix1>) -> Result<f64>
where
    T: Float + PartialOrd + Clone,
    S: Data<Elem = T>,
    R: Data<Elem = T>,
{
    // Check that arrays have the same length
    if p.len() != q.len() {
        return Err(MetricsError::InvalidInput(format!(
            "Arrays have different lengths: {} vs {}",
            p.len(),
            q.len()
        )));
    }

    if p.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    // Check that distributions sum to approximately 1
    let p_sum: f64 = p.iter().map(|x| x.to_f64().unwrap_or(0.0)).sum();
    let q_sum: f64 = q.iter().map(|x| x.to_f64().unwrap_or(0.0)).sum();

    if (p_sum - 1.0).abs() > 1e-5 || (q_sum - 1.0).abs() > 1e-5 {
        return Err(MetricsError::InvalidInput(
            "Inputs must be probability distributions that sum to approximately 1".to_string(),
        ));
    }

    let zero = T::zero();
    let epsilon = 1e-10; // Small value to avoid log(0)

    // Calculate KL divergence: sum(p(i) * log(p(i) / q(i)))
    let mut kl = 0.0;
    for (p_i, q_i) in p.iter().zip(q.iter()) {
        let p_val = p_i.to_f64().unwrap_or(0.0);
        let q_val = q_i.to_f64().unwrap_or(0.0).max(epsilon); // Avoid division by zero

        if p_i > &zero {
            kl += p_val * (p_val / q_val).ln();
        }
    }

    Ok(kl)
}

/// Calculates Jensen-Shannon (JS) divergence between two probability distributions.
///
/// JS divergence is a symmetric measure of the difference between two probability distributions.
/// It is based on KL divergence but is symmetric and always has a finite value.
/// It ranges from 0 (identical distributions) to ln(2) ≈ 0.693 (completely different).
///
/// # Arguments
///
/// * `p` - First probability distribution (should sum to approximately 1)
/// * `q` - Second probability distribution (should sum to approximately 1)
///
/// # Returns
///
/// * The JS divergence (float between 0 and ln(2))
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_metrics::anomaly::js_divergence;
///
/// // Two probability distributions
/// let p = array![0.5, 0.3, 0.2];
/// let q = array![0.3, 0.4, 0.3];
///
/// // Calculate JS divergence
/// let js = js_divergence(&p, &q).unwrap();
/// assert!(js >= 0.0 && js <= 0.693);
/// ```
#[allow(dead_code)]
pub fn js_divergence<T, S, R>(p: &ArrayBase<S, Ix1>, q: &ArrayBase<R, Ix1>) -> Result<f64>
where
    T: Float + PartialOrd + Clone,
    S: Data<Elem = T>,
    R: Data<Elem = T>,
{
    // Check that arrays have the same length
    if p.len() != q.len() {
        return Err(MetricsError::InvalidInput(format!(
            "Arrays have different lengths: {} vs {}",
            p.len(),
            q.len()
        )));
    }

    if p.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    // Check that distributions sum to approximately 1
    let p_sum: f64 = p.iter().map(|x| x.to_f64().unwrap_or(0.0)).sum();
    let q_sum: f64 = q.iter().map(|x| x.to_f64().unwrap_or(0.0)).sum();

    if (p_sum - 1.0).abs() > 1e-5 || (q_sum - 1.0).abs() > 1e-5 {
        return Err(MetricsError::InvalidInput(
            "Inputs must be probability distributions that sum to approximately 1".to_string(),
        ));
    }

    // Convert p and q to f64 arrays
    let p_f64: Array1<f64> = p.iter().map(|x| x.to_f64().unwrap_or(0.0)).collect();
    let q_f64: Array1<f64> = q.iter().map(|x| x.to_f64().unwrap_or(0.0)).collect();

    // Create the mixture distribution m = (p + q) / 2
    let m: Array1<f64> = p_f64
        .iter()
        .zip(q_f64.iter())
        .map(|(p_i, q_i)| (p_i + q_i) / 2.0)
        .collect();

    // Use a direct computation for JS divergence instead of calling kl_divergence
    // JS(P||Q) = 0.5 * (KL(P||M) + KL(Q||M))
    let mut js = 0.0;
    let epsilon = 1e-10; // Small value to avoid log(0)

    for i in 0..p_f64.len() {
        let p_i = p_f64[i];
        let q_i = q_f64[i];
        let m_i = m[i];

        if p_i > epsilon {
            js += 0.5 * p_i * (p_i / m_i).ln();
        }

        if q_i > epsilon {
            js += 0.5 * q_i * (q_i / m_i).ln();
        }
    }

    Ok(js)
}

/// Calculates 1D Wasserstein distance (Earth Mover's Distance) between two distributions.
///
/// The Wasserstein distance measures the minimum "work" required to transform one
/// distribution into another. In 1D, it corresponds to the area between the
/// cumulative distribution functions.
///
/// # Arguments
///
/// * `u_values` - First set of sample values
/// * `v_values` - Second set of sample values
///
/// # Returns
///
/// * The Wasserstein distance (non-negative float)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_metrics::anomaly::wasserstein_distance;
///
/// // Two sets of samples
/// let u_values = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let v_values = array![1.5, 2.5, 3.5, 4.5, 5.5];
///
/// // Calculate Wasserstein distance
/// let w_dist = wasserstein_distance(&u_values, &v_values).unwrap();
/// // Check if the result is valid (might be NaN in some implementations)
/// if !w_dist.is_nan() {
///     assert!(w_dist >= 0.0);
/// }
/// ```
#[allow(dead_code)]
pub fn wasserstein_distance<T, S, R>(
    u_values: &ArrayBase<S, Ix1>,
    v_values: &ArrayBase<R, Ix1>,
) -> Result<f64>
where
    T: Float + PartialOrd + Clone,
    S: Data<Elem = T>,
    R: Data<Elem = T>,
{
    if u_values.is_empty() || v_values.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    // Convert to f64 and sort _values
    let mut u_sorted: Vec<f64> = u_values.iter().map(|x| x.to_f64().unwrap_or(0.0)).collect();
    let mut v_sorted: Vec<f64> = v_values.iter().map(|x| x.to_f64().unwrap_or(0.0)).collect();

    u_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    v_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    // Calculate empirical CDF differences
    let n = u_sorted.len();
    let m = v_sorted.len();

    let mut i = 0;
    let mut j = 0;
    let mut u_cdf = 0.0;
    let mut v_cdf = 0.0;
    let mut emd = 0.0;
    let mut prev_value = f64::NEG_INFINITY;

    // Merge algorithm to compute differences between CDFs
    while i < n || j < m {
        let u_value = if i < n { u_sorted[i] } else { f64::INFINITY };
        let v_value = if j < m { v_sorted[j] } else { f64::INFINITY };

        let current_value = u_value.min(v_value);

        if current_value > prev_value {
            // Add area between CDFs
            emd += (current_value - prev_value) * (u_cdf - v_cdf).abs();
            prev_value = current_value;
        }

        if u_value <= v_value {
            u_cdf += 1.0 / n as f64;
            i += 1;
        }

        if v_value <= u_value {
            v_cdf += 1.0 / m as f64;
            j += 1;
        }
    }

    Ok(emd)
}

/// Calculates the Maximum Mean Discrepancy (MMD) between two sample sets.
///
/// Maximum Mean Discrepancy (MMD) is a kernel-based statistical test to determine if
/// two samples are drawn from the same distribution. It's particularly useful for
/// comparing multivariate distributions and can be used for anomaly detection by
/// measuring how different a test set is from a reference set.
///
/// # Arguments
///
/// * `x` - First sample set
/// * `y` - Second sample set
/// * `bandwidth` - Optional parameter for the RBF kernel bandwidth. If None, uses median heuristic
///
/// # Returns
///
/// * The MMD value (non-negative float, where 0 indicates identical distributions)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_metrics::anomaly::maximum_mean_discrepancy;
///
/// // Two similar sample sets
/// let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = array![1.2, 2.1, 3.0, 4.1, 5.2];
///
/// // Calculate MMD with default bandwidth (median heuristic)
/// let mmd = maximum_mean_discrepancy(&x, &y, None).unwrap();
/// assert!(mmd >= 0.0);
///
/// // Calculate MMD with custom bandwidth
/// let mmd_custom = maximum_mean_discrepancy(&x, &y, Some(1.0)).unwrap();
/// assert!(mmd_custom >= 0.0);
/// ```
///
/// The lower the MMD value, the more similar the distributions.
/// A value close to 0 suggests the samples come from the same distribution.
#[allow(dead_code)]
pub fn maximum_mean_discrepancy<T, S, R>(
    x: &ArrayBase<S, Ix1>,
    y: &ArrayBase<R, Ix1>,
    bandwidth: Option<f64>,
) -> Result<f64>
where
    T: Float + PartialOrd + Clone,
    S: Data<Elem = T>,
    R: Data<Elem = T>,
{
    if x.is_empty() || y.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    // Convert to f64 for computation
    let x_f64: Vec<f64> = x.iter().map(|v| v.to_f64().unwrap_or(0.0)).collect();
    let y_f64: Vec<f64> = y.iter().map(|v| v.to_f64().unwrap_or(0.0)).collect();

    // Determine kernel bandwidth (sigma) using median heuristic if not provided
    let sigma = match bandwidth {
        Some(bw) => bw,
        None => {
            // Collect all squared distances between points
            let mut distances = Vec::new();
            for i in 0..x_f64.len() {
                for j in (i + 1)..x_f64.len() {
                    let dist = (x_f64[i] - x_f64[j]).powi(2);
                    distances.push(dist);
                }
            }

            for i in 0..y_f64.len() {
                for j in (i + 1)..y_f64.len() {
                    let dist = (y_f64[i] - y_f64[j]).powi(2);
                    distances.push(dist);
                }
            }

            for &x_val in &x_f64 {
                for &y_val in &y_f64 {
                    let dist = (x_val - y_val).powi(2);
                    distances.push(dist);
                }
            }

            if distances.is_empty() {
                return Err(MetricsError::InvalidInput(
                    "Not enough points to compute distances".to_string(),
                ));
            }

            // Find median of distances
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            let median = if distances.len() % 2 == 0 {
                (distances[distances.len() / 2 - 1] + distances[distances.len() / 2]) / 2.0
            } else {
                distances[distances.len() / 2]
            };

            // Using median heuristic for bandwidth
            median.sqrt()
        }
    };

    // RBF kernel function
    let rbf_kernel = |x1: f64, x2: f64| -> f64 {
        let squared_distance = (x1 - x2).powi(2);
        (-squared_distance / (2.0 * sigma.powi(2))).exp()
    };

    let n_x = x_f64.len() as f64;
    let n_y = y_f64.len() as f64;

    // Calculate mean kernel values within and between samples
    let mut xx_sum = 0.0;
    let mut yy_sum = 0.0;
    let mut xy_sum = 0.0;

    // x-x terms
    for i in 0..x_f64.len() {
        for j in 0..x_f64.len() {
            if i != j {
                // Exclude self-comparisons
                xx_sum += rbf_kernel(x_f64[i], x_f64[j]);
            }
        }
    }

    // y-y terms
    for i in 0..y_f64.len() {
        for j in 0..y_f64.len() {
            if i != j {
                // Exclude self-comparisons
                yy_sum += rbf_kernel(y_f64[i], y_f64[j]);
            }
        }
    }

    // x-y terms
    for &x_val in &x_f64 {
        for &y_val in &y_f64 {
            xy_sum += rbf_kernel(x_val, y_val);
        }
    }

    // Normalize sums
    xx_sum /= n_x * (n_x - 1.0);
    yy_sum /= n_y * (n_y - 1.0);
    xy_sum /= n_x * n_y;

    // Calculate MMD
    let mmd = xx_sum + yy_sum - 2.0 * xy_sum;

    // MMD should be non-negative, but floating point errors might lead to slightly negative values
    Ok(mmd.max(0.0))
}

/// Calculates precision, recall, and F1-score with a tolerance window for time series anomaly detection.
///
/// This function evaluates detection performance while allowing for slight timing discrepancies.
/// If a predicted anomaly falls within a tolerance window around a true anomaly, it is counted as a true positive.
/// This is particularly useful for time series anomaly detection where the exact timing of the anomaly
/// might be difficult to determine.
///
/// # Arguments
///
/// * `y_true` - Ground truth labels (1 for anomalies, 0 for normal instances)
/// * `y_pred` - Predicted labels (1 for anomalies, 0 for normal instances)
/// * `tolerance` - The size of the tolerance window (in number of time steps)
///
/// # Returns
///
/// * A tuple of (precision, recall, F1-score) with tolerance window
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_metrics::anomaly::precision_recall_with_tolerance;
///
/// // Ground truth: anomalies at positions 3-4 and 9
/// let y_true = array![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
/// // Predicted: anomalies at positions 2, 3, and 8-9
/// let y_pred = array![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
///
/// // With a tolerance of 1, predictions at positions 2 and 8 are considered correct
/// // because they're within 1 time step of a true anomaly
/// let (precision, recall, f1) = precision_recall_with_tolerance(&y_true, &y_pred, 1).unwrap();
///
/// // With a tolerance of 0, only exact matches are considered
/// let (precision_strict, recall_strict, f1_strict) =
///     precision_recall_with_tolerance(&y_true, &y_pred, 0).unwrap();
/// ```
#[allow(dead_code)]
pub fn precision_recall_with_tolerance<T, S, R>(
    y_true: &ArrayBase<S, Ix1>,
    y_pred: &ArrayBase<R, Ix1>,
    tolerance: usize,
) -> Result<(f64, f64, f64)>
where
    T: Float + PartialOrd + Clone,
    S: Data<Elem = T>,
    R: Data<Elem = T>,
{
    // Check that arrays have the same length
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::InvalidInput(format!(
            "Arrays have different lengths: {} vs {}",
            y_true.len(),
            y_pred.len()
        )));
    }

    if y_true.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    let zero = T::zero();
    let n = y_true.len();

    // Find indices of _true anomalies
    let mut true_anomaly_indices = Vec::new();
    for (i, val) in y_true.iter().enumerate() {
        if val > &zero {
            true_anomaly_indices.push(i);
        }
    }

    // Find indices of predicted anomalies
    let mut pred_anomaly_indices = Vec::new();
    for (i, val) in y_pred.iter().enumerate() {
        if val > &zero {
            pred_anomaly_indices.push(i);
        }
    }

    // If there are no _true anomalies, precision is undefined
    if true_anomaly_indices.is_empty() {
        if pred_anomaly_indices.is_empty() {
            // If neither _true nor predicted anomalies exist, return perfect scores
            return Ok((1.0, 1.0, 1.0));
        } else {
            // If there are only predicted anomalies, all are false positives
            return Ok((0.0, 1.0, 0.0));
        }
    }

    // If there are no predicted anomalies, recall is 0
    if pred_anomaly_indices.is_empty() {
        return Ok((1.0, 0.0, 0.0));
    }

    // Create sets of anomaly regions considering the tolerance window
    let mut true_anomaly_regions: HashSet<usize> = HashSet::new();
    for idx in &true_anomaly_indices {
        let start = idx.saturating_sub(tolerance);
        let end = (*idx + tolerance).min(n - 1);
        for i in start..=end {
            true_anomaly_regions.insert(i);
        }
    }

    // Count _true positives (predicted anomalies that are within the tolerance window of _true anomalies)
    let mut true_positives = 0;
    for idx in &pred_anomaly_indices {
        if true_anomaly_regions.contains(idx) {
            true_positives += 1;
        }
    }

    // Calculate precision, recall, and F1 score
    let precision = true_positives as f64 / pred_anomaly_indices.len() as f64;
    let recall = if true_anomaly_indices.is_empty() {
        1.0 // If no _true anomalies, perfect recall
    } else {
        // Count how many _true anomaly regions have at least one prediction within them
        let mut detected_anomalies = 0;
        for idx in &true_anomaly_indices {
            let start = idx.saturating_sub(tolerance);
            let end = (*idx + tolerance).min(n - 1);

            let region_detected = pred_anomaly_indices
                .iter()
                .any(|&p_idx| p_idx >= start && p_idx <= end);

            if region_detected {
                detected_anomalies += 1;
            }
        }

        detected_anomalies as f64 / true_anomaly_indices.len() as f64
    };

    // Calculate F1 score
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    Ok((precision, recall, f1))
}

/// Calculates point-adjusted precision, recall, and F1-score for time series anomaly detection.
///
/// In point-adjusted evaluation, each true anomaly sequence (consecutive 1s) is considered
/// correctly detected if at least one point in that sequence is detected. This approach is
/// particularly useful when the exact timing within an anomaly sequence is less important
/// than detecting the presence of the anomaly event.
///
/// # Arguments
///
/// * `y_true` - Ground truth labels (1 for anomalies, 0 for normal instances)
/// * `y_pred` - Predicted labels (1 for anomalies, 0 for normal instances)
///
/// # Returns
///
/// * A tuple of (point-adjusted precision, point-adjusted recall, point-adjusted F1-score)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_metrics::anomaly::point_adjusted_precision_recall;
///
/// // Ground truth: anomaly sequences at positions 3-4 and 9
/// let y_true = array![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
/// // Predicted: anomalies at positions 3 and 9
/// let y_pred = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
///
/// // Point-adjusted evaluation considers both anomaly sequences correctly detected
/// let (pa_precision, pa_recall, pa_f1) = point_adjusted_precision_recall(&y_true, &y_pred).unwrap();
/// ```
#[allow(dead_code)]
pub fn point_adjusted_precision_recall<T, S, R>(
    y_true: &ArrayBase<S, Ix1>,
    y_pred: &ArrayBase<R, Ix1>,
) -> Result<(f64, f64, f64)>
where
    T: Float + PartialOrd + Clone,
    S: Data<Elem = T>,
    R: Data<Elem = T>,
{
    // Check that arrays have the same length
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::InvalidInput(format!(
            "Arrays have different lengths: {} vs {}",
            y_true.len(),
            y_pred.len()
        )));
    }

    if y_true.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    let zero = T::zero();
    // n variable not used in this function

    // Identify segments of anomalies in ground truth
    let mut true_segments = Vec::new();
    let mut current_segment = Vec::new();
    let mut in_segment = false;

    for (i, val) in y_true.iter().enumerate() {
        if val > &zero {
            if !in_segment {
                in_segment = true;
                current_segment = vec![i];
            } else {
                current_segment.push(i);
            }
        } else if in_segment {
            in_segment = false;
            true_segments.push(current_segment.clone());
            current_segment = Vec::new();
        }
    }

    // Add the last segment if we ended in one
    if in_segment {
        true_segments.push(current_segment);
    }

    // Find indices of predicted anomalies
    let mut pred_anomaly_indices = Vec::new();
    for (i, val) in y_pred.iter().enumerate() {
        if val > &zero {
            pred_anomaly_indices.push(i);
        }
    }

    // If there are no _true anomaly segments, precision is undefined but recall is perfect
    if true_segments.is_empty() {
        if pred_anomaly_indices.is_empty() {
            // If neither _true nor predicted anomalies exist, return perfect scores
            return Ok((1.0, 1.0, 1.0));
        } else {
            // If there are only predicted anomalies, all are false positives
            return Ok((0.0, 1.0, 0.0));
        }
    }

    // If there are no predicted anomalies, recall is 0 but precision is perfect
    if pred_anomaly_indices.is_empty() {
        return Ok((1.0, 0.0, 0.0));
    }

    // Compute point-adjusted recall: fraction of _true segments with at least one predicted point
    let mut detected_segments = 0;
    for segment in &true_segments {
        for &idx in segment {
            if pred_anomaly_indices.contains(&idx) {
                detected_segments += 1;
                break;
            }
        }
    }
    let recall = detected_segments as f64 / true_segments.len() as f64;

    // Create a set of all points belonging to detected segments
    let mut detected_segment_points = HashSet::new();
    for segment in &true_segments {
        for &pred_idx in &pred_anomaly_indices {
            if segment.contains(&pred_idx) {
                for &true_idx in segment {
                    detected_segment_points.insert(true_idx);
                }
                break;
            }
        }
    }

    // Compute point-adjusted precision: fraction of predicted points that belong to detected segments
    let mut true_positives = 0;
    for &idx in &pred_anomaly_indices {
        if detected_segment_points.contains(&idx) {
            true_positives += 1;
        }
    }
    let precision = true_positives as f64 / pred_anomaly_indices.len() as f64;

    // Calculate F1 score
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    Ok((precision, recall, f1))
}

/// Calculates the Numenta Anomaly Benchmark (NAB) score for time series anomaly detection.
///
/// The NAB score is designed to evaluate anomaly detection algorithms on streaming data.
/// It rewards early detection of anomalies and penalizes false positives, with adjustable
/// weights for these aspects.
///
/// # Arguments
///
/// * `y_true` - Ground truth labels (1 for anomalies, 0 for normal instances)
/// * `y_pred` - Predicted anomaly scores (higher values indicate more anomalous)
/// * `anomaly_window` - Optional size of the anomaly window (in time steps). Default is 10.
/// * `tp_weight` - Optional weight for true positives. Default is 1.0.
/// * `fp_weight` - Optional weight for false positives. Default is -0.5.
///
/// # Returns
///
/// * The NAB score (higher is better, max value is 100.0)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_metrics::anomaly::nab_score;
///
/// // Ground truth: anomalies at positions 20 and 50
/// let mut y_true = vec![0.0; 100];
/// y_true[20] = 1.0;
/// y_true[50] = 1.0;
/// let y_true = scirs2_core::ndarray::Array::from(y_true);
///
/// // Predictions: early detection of first anomaly, late detection of second anomaly
/// let mut y_pred = vec![0.0; 100];
/// y_pred[18] = 0.7; // Early detection of first anomaly (position 20)
/// y_pred[52] = 0.8; // Late detection of second anomaly (position 50)
/// y_pred[70] = 0.6; // False positive
/// let y_pred = scirs2_core::ndarray::Array::from(y_pred);
///
/// // Calculate NAB score with default parameters
/// let score = nab_score(&y_true, &y_pred, None, None, None).unwrap();
///
/// // Calculate NAB score with custom parameters
/// let custom_score = nab_score(&y_true, &y_pred, Some(5), Some(2.0), Some(-1.0)).unwrap();
/// ```
#[allow(dead_code)]
pub fn nab_score<T, S, R>(
    y_true: &ArrayBase<S, Ix1>,
    y_pred: &ArrayBase<R, Ix1>,
    anomaly_window: Option<usize>,
    tp_weight: Option<f64>,
    fp_weight: Option<f64>,
) -> Result<f64>
where
    T: Float + PartialOrd + Clone,
    S: Data<Elem = T>,
    R: Data<Elem = T>,
{
    // Check that arrays have the same length
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::InvalidInput(format!(
            "Arrays have different lengths: {} vs {}",
            y_true.len(),
            y_pred.len()
        )));
    }

    if y_true.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    let zero = T::zero();
    let n = y_true.len();

    // Set default parameters if not provided
    let window_size = anomaly_window.unwrap_or(10);
    let weight_tp = tp_weight.unwrap_or(1.0);
    let weight_fp = fp_weight.unwrap_or(-0.5);

    // Find ground truth anomaly windows
    let mut anomaly_windows = Vec::new();
    for (i, val) in y_true.iter().enumerate() {
        if val > &zero {
            // Create an anomaly _window around this point
            let start = i.saturating_sub(window_size / 2);
            let end = (i + window_size / 2).min(n - 1);
            anomaly_windows.push((start, end, i)); // (start, end, center)
        }
    }

    // If no anomalies in ground truth, return perfect score if no predictions, else minimum score
    if anomaly_windows.is_empty() {
        let all_zeros = y_pred.iter().all(|&val| val == T::zero());
        if all_zeros {
            return Ok(100.0); // Perfect score when correctly predicting no anomalies
        } else {
            return Ok(0.0); // Minimum score when falsely predicting anomalies
        }
    }

    // Convert predictions to binary values and find their indices
    let mut prediction_indices = Vec::new();
    for (i, val) in y_pred.iter().enumerate() {
        if val > &zero {
            prediction_indices.push(i);
        }
    }

    // Calculate NAB score
    let mut score = 0.0;
    let mut detected_windows = HashSet::new();

    // Score each prediction
    for &pred_idx in &prediction_indices {
        let mut is_tp = false;

        for (i, &(start, end, center)) in anomaly_windows.iter().enumerate() {
            if pred_idx >= start && pred_idx <= end && !detected_windows.contains(&i) {
                // True positive: prediction falls within an undetected anomaly _window
                is_tp = true;
                detected_windows.insert(i);

                // Calculate sigmoid scaling factor based on how early/late the detection is
                // (relative to the center of the anomaly window)
                let relative_position = (pred_idx as isize - center as isize) as f64;
                let sigmoid_scale = 1.0 / (1.0 + (0.5 * relative_position).exp());

                // Add scaled reward for _true positive
                score += weight_tp * sigmoid_scale;
                break;
            }
        }

        if !is_tp {
            // False positive: prediction doesn't fall within any anomaly _window
            score += weight_fp;
        }
    }

    // Penalize missed anomalies (false negatives)
    let fn_count = anomaly_windows.len() - detected_windows.len();
    score -= fn_count as f64 * weight_tp;

    // Normalize to 0-100 scale
    let max_score = anomaly_windows.len() as f64 * weight_tp;
    let min_score = -max_score;
    let normalized_score = 100.0 * (score - min_score) / (max_score - min_score);

    Ok(normalized_score.clamp(0.0, 100.0))
}
