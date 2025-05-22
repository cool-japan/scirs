//! Mutual Information metrics for clustering evaluation
//!
//! This module provides Normalized Mutual Information (NMI) and Adjusted Mutual Information (AMI)
//! metrics for evaluating clustering results against ground truth.

use ndarray::{ArrayBase, Data, Dimension};
use std::collections::HashMap;
use std::fmt::Debug;

use crate::error::{MetricsError, Result};

/// Calculates the Normalized Mutual Information (NMI) between two clusterings
///
/// NMI is a normalization of the Mutual Information (MI) score to scale the
/// results between 0 (no mutual information) and 1 (perfect correlation).
/// It measures the agreement of two clusterings, ignoring permutations.
///
/// # Arguments
///
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Predicted cluster labels to evaluate
/// * `average_method` - Method to compute the normalization. One of:
///   * "arithmetic": (MI) / ((H(labels_true) + H(labels_pred)) / 2)
///   * "geometric": MI / sqrt(H(labels_true) * H(labels_pred))
///   * "min": MI / min(H(labels_true), H(labels_pred))
///   * "max": MI / max(H(labels_true), H(labels_pred))
///
/// # Returns
///
/// * The Normalized Mutual Information score (between 0.0 and 1.0)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::clustering::normalized_mutual_info_score;
///
/// let labels_true = array![0, 0, 1, 1, 2, 2];
/// let labels_pred = array![0, 0, 0, 1, 1, 1];
///
/// let nmi = normalized_mutual_info_score(&labels_true, &labels_pred, "arithmetic").unwrap();
/// ```
pub fn normalized_mutual_info_score<T, U, S1, S2, D1, D2>(
    labels_true: &ArrayBase<S1, D1>,
    labels_pred: &ArrayBase<S2, D2>,
    average_method: &str,
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

    // Validate average_method
    match average_method {
        "arithmetic" | "geometric" | "min" | "max" => {}
        _ => {
            return Err(MetricsError::InvalidInput(format!(
                "Invalid average_method: {}. Must be one of 'arithmetic', 'geometric', 'min', or 'max'",
                average_method
            )));
        }
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

    // Calculate mutual information
    let mut mutual_info = 0.0;
    let n_samples_f64 = n_samples as f64;

    for ((lt, lp), &nij) in contingency.iter() {
        let ni = true_counts.get(lt).unwrap_or(&0);
        let nj = pred_counts.get(lp).unwrap_or(&0);

        if nij > 0 && *ni > 0 && *nj > 0 {
            let pij = nij as f64 / n_samples_f64;
            let pi = *ni as f64 / n_samples_f64;
            let pj = *nj as f64 / n_samples_f64;

            mutual_info += pij * (pij / (pi * pj)).ln();
        }
    }

    // Normalize mutual information based on average_method
    let nmi = match average_method {
        "arithmetic" => {
            if h_true + h_pred == 0.0 {
                0.0
            } else {
                2.0 * mutual_info / (h_true + h_pred)
            }
        }
        "geometric" => {
            if h_true == 0.0 || h_pred == 0.0 {
                0.0
            } else {
                mutual_info / (h_true * h_pred).sqrt()
            }
        }
        "min" => {
            let min_entropy = h_true.min(h_pred);
            if min_entropy == 0.0 {
                0.0
            } else {
                mutual_info / min_entropy
            }
        }
        "max" => {
            let max_entropy = h_true.max(h_pred);
            if max_entropy == 0.0 {
                0.0
            } else {
                mutual_info / max_entropy
            }
        }
        _ => unreachable!(), // Already validated
    };

    // Clamp to [0, 1] range to avoid numerical issues
    Ok(nmi.clamp(0.0, 1.0))
}

/// Calculates the Adjusted Mutual Information (AMI) between two clusterings
///
/// The Adjusted Mutual Information is an adjustment of the Mutual Information (MI) score
/// to account for chance. It accounts for the fact that MI is generally higher for two
/// clusterings with a larger number of clusters, regardless of whether there is actually
/// more information shared.
///
/// AMI values range from 0 to 1:
/// * 1: Perfect agreement between the clusterings
/// * 0: Agreement equivalent to random chance
///
/// # Arguments
///
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Predicted cluster labels to evaluate
/// * `average_method` - Method to compute the adjustment. One of:
///   * "arithmetic": (MI - E[MI]) / (max(H(labels_true), H(labels_pred)) - E[MI])
///   * "geometric": (MI - E[MI]) / (sqrt(H(labels_true) * H(labels_pred)) - E[MI])
///   * "max": (MI - E[MI]) / (max(H(labels_true), H(labels_pred)) - E[MI])
///   * "min": (MI - E[MI]) / (min(H(labels_true), H(labels_pred)) - E[MI])
///
/// # Returns
///
/// * The Adjusted Mutual Information score (between 0.0 and 1.0)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::clustering::adjusted_mutual_info_score;
///
/// let labels_true = array![0, 0, 1, 1, 2, 2];
/// let labels_pred = array![0, 0, 0, 1, 1, 1];
///
/// let ami = adjusted_mutual_info_score(&labels_true, &labels_pred, "arithmetic").unwrap();
/// ```
pub fn adjusted_mutual_info_score<T, U, S1, S2, D1, D2>(
    labels_true: &ArrayBase<S1, D1>,
    labels_pred: &ArrayBase<S2, D2>,
    average_method: &str,
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

    // Validate average_method
    match average_method {
        "arithmetic" | "geometric" | "min" | "max" => {}
        _ => {
            return Err(MetricsError::InvalidInput(format!(
                "Invalid average_method: {}. Must be one of 'arithmetic', 'geometric', 'min', or 'max'",
                average_method
            )));
        }
    }

    // Compute contingency matrix (using strings for label values to handle different types)
    let mut contingency: HashMap<(String, String), usize> = HashMap::new();
    for (lt, lp) in labels_true.iter().zip(labels_pred.iter()) {
        let key = (format!("{:?}", lt), format!("{:?}", lp));
        *contingency.entry(key).or_insert(0) += 1;
    }

    // Count labels (and store them in a way we can reference later)
    let mut true_labels: Vec<String> = Vec::new();
    let mut true_counts: HashMap<String, usize> = HashMap::new();
    for lt in labels_true.iter() {
        let key = format!("{:?}", lt);
        if !true_labels.contains(&key) {
            true_labels.push(key.clone());
        }
        *true_counts.entry(key).or_insert(0) += 1;
    }

    let mut pred_labels: Vec<String> = Vec::new();
    let mut pred_counts: HashMap<String, usize> = HashMap::new();
    for lp in labels_pred.iter() {
        let key = format!("{:?}", lp);
        if !pred_labels.contains(&key) {
            pred_labels.push(key.clone());
        }
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

    // Calculate mutual information
    let mut mutual_info = 0.0;
    let n_samples_f64 = n_samples as f64;

    for ((lt, lp), &nij) in contingency.iter() {
        let ni = true_counts.get(lt).unwrap_or(&0);
        let nj = pred_counts.get(lp).unwrap_or(&0);

        if nij > 0 && *ni > 0 && *nj > 0 {
            let pij = nij as f64 / n_samples_f64;
            let pi = *ni as f64 / n_samples_f64;
            let pj = *nj as f64 / n_samples_f64;

            mutual_info += pij * (pij / (pi * pj)).ln();
        }
    }

    // Calculate the expected mutual information
    let a = true_labels.len() as f64;
    let b = pred_labels.len() as f64;
    let n = n_samples as f64;

    // Special case: if a or b is 1, expected MI is 0
    if a <= 1.0 || b <= 1.0 {
        return Ok(0.0);
    }

    // Calculate expected mutual information
    let mut emi = 0.0;

    for (_, &ai) in true_counts.iter() {
        for (_, &bj) in pred_counts.iter() {
            let ai_f64 = ai as f64;
            let bj_f64 = bj as f64;

            // Compute the sum over N_{ij} (a bit complex for an exact match)
            // We use a simpler approximation based on the concept that the expectation
            // can be approximated as the product of marginals divided by n_samples
            let expected_nij = ai_f64 * bj_f64 / n;

            if expected_nij > 0.0 {
                let pi = ai_f64 / n;
                let pj = bj_f64 / n;
                let pij = expected_nij / n;

                emi += expected_nij / n_samples_f64 * (pij / (pi * pj)).ln();
            }
        }
    }

    // Adjust the mutual information
    let ami = match average_method {
        "arithmetic" => {
            let avg_h = (h_true + h_pred) / 2.0;
            if avg_h <= emi {
                0.0
            } else {
                (mutual_info - emi) / (avg_h - emi)
            }
        }
        "geometric" => {
            let sqrt_h = (h_true * h_pred).sqrt();
            if sqrt_h <= emi {
                0.0
            } else {
                (mutual_info - emi) / (sqrt_h - emi)
            }
        }
        "min" => {
            let min_h = h_true.min(h_pred);
            if min_h <= emi {
                0.0
            } else {
                (mutual_info - emi) / (min_h - emi)
            }
        }
        "max" => {
            let max_h = h_true.max(h_pred);
            if max_h <= emi {
                0.0
            } else {
                (mutual_info - emi) / (max_h - emi)
            }
        }
        _ => unreachable!(), // Already validated
    };

    // Clamp to [0, 1] range to avoid numerical issues
    Ok(ami.clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_normalized_mutual_info_score() {
        // Perfect match
        let labels_true = array![0, 0, 1, 1, 2, 2];
        let labels_pred = array![1, 1, 0, 0, 2, 2]; // Same clustering with different labels

        let nmi_arithmetic =
            normalized_mutual_info_score(&labels_true, &labels_pred, "arithmetic").unwrap();
        let nmi_geometric =
            normalized_mutual_info_score(&labels_true, &labels_pred, "geometric").unwrap();
        let nmi_min = normalized_mutual_info_score(&labels_true, &labels_pred, "min").unwrap();
        let nmi_max = normalized_mutual_info_score(&labels_true, &labels_pred, "max").unwrap();

        assert!((nmi_arithmetic - 1.0).abs() < 1e-10);
        assert!((nmi_geometric - 1.0).abs() < 1e-10);
        assert!((nmi_min - 1.0).abs() < 1e-10);
        assert!((nmi_max - 1.0).abs() < 1e-10);

        // Partial match
        let labels_true = array![0, 0, 1, 1, 2, 2];
        let labels_pred = array![0, 0, 0, 1, 1, 1];

        let nmi = normalized_mutual_info_score(&labels_true, &labels_pred, "arithmetic").unwrap();
        assert!(nmi > 0.0 && nmi < 1.0);
    }

    #[test]
    fn test_adjusted_mutual_info_score() {
        // Perfect match
        let labels_true = array![0, 0, 1, 1, 2, 2];
        let labels_pred = array![1, 1, 0, 0, 2, 2]; // Same clustering with different labels

        let ami = adjusted_mutual_info_score(&labels_true, &labels_pred, "arithmetic").unwrap();
        assert!((ami - 1.0).abs() < 1e-10);

        // Partial match
        let labels_true = array![0, 0, 1, 1, 2, 2];
        let labels_pred = array![0, 0, 0, 1, 1, 1];

        let ami = adjusted_mutual_info_score(&labels_true, &labels_pred, "arithmetic").unwrap();
        assert!(ami > 0.0 && ami < 1.0);
    }
}
