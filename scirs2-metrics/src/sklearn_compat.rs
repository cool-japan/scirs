//! Scikit-learn compatibility module
//!
//! This module provides implementations of metrics that are equivalent to
//! those found in scikit-learn, ensuring API compatibility and identical
//! results where possible.

use crate::error::{MetricsError, Result};
use ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;
use std::collections::HashSet;

/// Type alias for precision, recall, fscore, support tuple
type PrecisionRecallFscoreSupport = (Array1<f64>, Array1<f64>, Array1<f64>, Array1<usize>);

/// Equivalent to sklearn.metrics.classification_report
///
/// Build a text report showing the main classification metrics.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) target values
/// * `y_pred` - Estimated targets as returned by a classifier  
/// * `labels` - Optional list of label indices to include in the report
/// * `target_names` - Optional display names matching the labels (same order)
/// * `sample_weight` - Optional sample weights
/// * `digits` - Number of digits for formatting output floating point values
/// * `output_dict` - If True, return output as dict instead of string
/// * `zero_division` - Sets the value to return when there is a zero division
///
/// # Returns
///
/// * Text summary of precision, recall, f1-score for each class (if output_dict=False)
/// * Dictionary with precision, recall, f1-score for each class (if output_dict=True)
#[derive(Debug, Clone)]
pub struct ClassificationReport {
    pub precision: HashMap<String, f64>,
    pub recall: HashMap<String, f64>,
    pub f1_score: HashMap<String, f64>,
    pub support: HashMap<String, usize>,
    pub accuracy: f64,
    pub macro_avg: ClassificationMetrics,
    pub weighted_avg: ClassificationMetrics,
}

#[derive(Debug, Clone)]
pub struct ClassificationMetrics {
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub support: usize,
}

/// Equivalent to sklearn.metrics.classification_report
pub fn classification_report_sklearn(
    y_true: &Array1<i32>,
    y_pred: &Array1<i32>,
    labels: Option<&[i32]>,
    target_names: Option<&[String]>,
    _digits: usize,
    zero_division: f64,
) -> Result<ClassificationReport> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::InvalidInput(
            "y_true and y_pred must have the same length".to_string(),
        ));
    }

    // Get unique labels
    let unique_labels: Vec<i32> = if let Some(labels) = labels {
        labels.to_vec()
    } else {
        let all_labels: HashSet<i32> = y_true.iter().chain(y_pred.iter()).copied().collect();
        let mut sorted_labels: Vec<i32> = all_labels.into_iter().collect();
        sorted_labels.sort();
        sorted_labels
    };

    // Calculate per-class metrics
    let mut precision_map = HashMap::new();
    let mut recall_map = HashMap::new();
    let mut f1_map = HashMap::new();
    let mut support_map = HashMap::new();

    for &label in &unique_labels {
        let (precision, recall, f1, support) =
            calculate_class_metrics(y_true, y_pred, label, zero_division)?;

        let label_name = if let Some(names) = target_names {
            if let Some(pos) = unique_labels.iter().position(|&x| x == label) {
                if pos < names.len() {
                    names[pos].clone()
                } else {
                    label.to_string()
                }
            } else {
                label.to_string()
            }
        } else {
            label.to_string()
        };

        precision_map.insert(label_name.clone(), precision);
        recall_map.insert(label_name.clone(), recall);
        f1_map.insert(label_name.clone(), f1);
        support_map.insert(label_name, support);
    }

    // Calculate accuracy
    let accuracy = accuracy_score_sklearn(y_true, y_pred)?;

    // Calculate macro averages
    let macro_precision = precision_map.values().sum::<f64>() / precision_map.len() as f64;
    let macro_recall = recall_map.values().sum::<f64>() / recall_map.len() as f64;
    let macro_f1 = f1_map.values().sum::<f64>() / f1_map.len() as f64;
    let macro_support = support_map.values().sum::<usize>();

    // Calculate weighted averages
    let total_support = support_map.values().sum::<usize>() as f64;
    let weighted_precision = precision_map
        .iter()
        .zip(support_map.iter())
        .map(
            |((label1, &p), (label2, &s))| {
                if label1 == label2 {
                    p * s as f64
                } else {
                    0.0
                }
            },
        )
        .sum::<f64>()
        / total_support;

    let weighted_recall = recall_map
        .iter()
        .zip(support_map.iter())
        .map(
            |((label1, &r), (label2, &s))| {
                if label1 == label2 {
                    r * s as f64
                } else {
                    0.0
                }
            },
        )
        .sum::<f64>()
        / total_support;

    let weighted_f1 = f1_map
        .iter()
        .zip(support_map.iter())
        .map(
            |((label1, &f), (label2, &s))| {
                if label1 == label2 {
                    f * s as f64
                } else {
                    0.0
                }
            },
        )
        .sum::<f64>()
        / total_support;

    Ok(ClassificationReport {
        precision: precision_map,
        recall: recall_map,
        f1_score: f1_map,
        support: support_map,
        accuracy,
        macro_avg: ClassificationMetrics {
            precision: macro_precision,
            recall: macro_recall,
            f1_score: macro_f1,
            support: macro_support,
        },
        weighted_avg: ClassificationMetrics {
            precision: weighted_precision,
            recall: weighted_recall,
            f1_score: weighted_f1,
            support: macro_support,
        },
    })
}

/// Calculate metrics for a specific class
fn calculate_class_metrics(
    y_true: &Array1<i32>,
    y_pred: &Array1<i32>,
    target_class: i32,
    zero_division: f64,
) -> Result<(f64, f64, f64, usize)> {
    let mut tp = 0;
    let mut fp = 0;
    let mut fn_count = 0;
    let mut support = 0;

    for (&true_val, &pred_val) in y_true.iter().zip(y_pred.iter()) {
        if true_val == target_class {
            support += 1;
            if pred_val == target_class {
                tp += 1;
            } else {
                fn_count += 1;
            }
        } else if pred_val == target_class {
            fp += 1;
        }
    }

    let precision = if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        zero_division
    };

    let recall = if tp + fn_count > 0 {
        tp as f64 / (tp + fn_count) as f64
    } else {
        zero_division
    };

    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        zero_division
    };

    Ok((precision, recall, f1, support))
}

/// Equivalent to sklearn.metrics.accuracy_score
pub fn accuracy_score_sklearn(y_true: &Array1<i32>, y_pred: &Array1<i32>) -> Result<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::InvalidInput(
            "y_true and y_pred must have the same length".to_string(),
        ));
    }

    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(&true_val, &pred_val)| true_val == pred_val)
        .count();

    Ok(correct as f64 / y_true.len() as f64)
}

/// Equivalent to sklearn.metrics.precision_recall_fscore_support
pub fn precision_recall_fscore_support_sklearn(
    y_true: &Array1<i32>,
    y_pred: &Array1<i32>,
    beta: f64,
    labels: Option<&[i32]>,
    _pos_label: Option<i32>,
    average: Option<&str>,
    _warn_for: Option<&[&str]>,
    zero_division: f64,
) -> Result<PrecisionRecallFscoreSupport> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::InvalidInput(
            "y_true and y_pred must have the same length".to_string(),
        ));
    }

    // Determine labels to use
    let target_labels: Vec<i32> = if let Some(labels) = labels {
        labels.to_vec()
    } else {
        let all_labels: HashSet<i32> = y_true.iter().chain(y_pred.iter()).copied().collect();
        let mut sorted_labels: Vec<i32> = all_labels.into_iter().collect();
        sorted_labels.sort();
        sorted_labels
    };

    let mut precisions = Vec::new();
    let mut recalls = Vec::new();
    let mut fscores = Vec::new();
    let mut supports = Vec::new();

    for &label in &target_labels {
        let (precision, recall, _f1, support) =
            calculate_class_metrics(y_true, y_pred, label, zero_division)?;

        // Calculate F-beta score
        let fbeta = if precision + recall > 0.0 {
            (1.0 + beta * beta) * precision * recall / (beta * beta * precision + recall)
        } else {
            zero_division
        };

        precisions.push(precision);
        recalls.push(recall);
        fscores.push(fbeta);
        supports.push(support);
    }

    // Handle averaging
    if let Some(avg_type) = average {
        match avg_type {
            "micro" => {
                let (micro_precision, micro_recall, micro_fbeta, total_support) =
                    calculate_micro_average(y_true, y_pred, beta, &target_labels, zero_division)?;
                Ok((
                    Array1::from_vec(vec![micro_precision]),
                    Array1::from_vec(vec![micro_recall]),
                    Array1::from_vec(vec![micro_fbeta]),
                    Array1::from_vec(vec![total_support]),
                ))
            }
            "macro" => {
                let macro_precision = precisions.iter().sum::<f64>() / precisions.len() as f64;
                let macro_recall = recalls.iter().sum::<f64>() / recalls.len() as f64;
                let macro_fbeta = fscores.iter().sum::<f64>() / fscores.len() as f64;
                let total_support = supports.iter().sum::<usize>();
                Ok((
                    Array1::from_vec(vec![macro_precision]),
                    Array1::from_vec(vec![macro_recall]),
                    Array1::from_vec(vec![macro_fbeta]),
                    Array1::from_vec(vec![total_support]),
                ))
            }
            "weighted" => {
                let total_support = supports.iter().sum::<usize>() as f64;
                let weighted_precision = precisions
                    .iter()
                    .zip(supports.iter())
                    .map(|(&p, &s)| p * s as f64)
                    .sum::<f64>()
                    / total_support;
                let weighted_recall = recalls
                    .iter()
                    .zip(supports.iter())
                    .map(|(&r, &s)| r * s as f64)
                    .sum::<f64>()
                    / total_support;
                let weighted_fbeta = fscores
                    .iter()
                    .zip(supports.iter())
                    .map(|(&f, &s)| f * s as f64)
                    .sum::<f64>()
                    / total_support;
                Ok((
                    Array1::from_vec(vec![weighted_precision]),
                    Array1::from_vec(vec![weighted_recall]),
                    Array1::from_vec(vec![weighted_fbeta]),
                    Array1::from_vec(vec![total_support as usize]),
                ))
            }
            _ => Err(MetricsError::InvalidInput(format!(
                "Unsupported average type: {}",
                avg_type
            ))),
        }
    } else {
        Ok((
            Array1::from_vec(precisions),
            Array1::from_vec(recalls),
            Array1::from_vec(fscores),
            Array1::from_vec(supports),
        ))
    }
}

/// Calculate micro-averaged metrics
fn calculate_micro_average(
    y_true: &Array1<i32>,
    y_pred: &Array1<i32>,
    beta: f64,
    labels: &[i32],
    zero_division: f64,
) -> Result<(f64, f64, f64, usize)> {
    let mut total_tp = 0;
    let mut total_fp = 0;
    let mut total_fn = 0;
    let mut total_support = 0;

    for &label in labels {
        let mut tp = 0;
        let mut fp = 0;
        let mut fn_count = 0;

        for (&true_val, &pred_val) in y_true.iter().zip(y_pred.iter()) {
            if true_val == label {
                total_support += 1;
                if pred_val == label {
                    tp += 1;
                } else {
                    fn_count += 1;
                }
            } else if pred_val == label {
                fp += 1;
            }
        }

        total_tp += tp;
        total_fp += fp;
        total_fn += fn_count;
    }

    let micro_precision = if total_tp + total_fp > 0 {
        total_tp as f64 / (total_tp + total_fp) as f64
    } else {
        zero_division
    };

    let micro_recall = if total_tp + total_fn > 0 {
        total_tp as f64 / (total_tp + total_fn) as f64
    } else {
        zero_division
    };

    let micro_fbeta = if micro_precision + micro_recall > 0.0 {
        (1.0 + beta * beta) * micro_precision * micro_recall
            / (beta * beta * micro_precision + micro_recall)
    } else {
        zero_division
    };

    Ok((micro_precision, micro_recall, micro_fbeta, total_support))
}

/// Equivalent to sklearn.metrics.multilabel_confusion_matrix
pub fn multilabel_confusion_matrix_sklearn(
    y_true: &Array2<i32>,
    y_pred: &Array2<i32>,
    sample_weight: Option<&Array1<f64>>,
    labels: Option<&[usize]>,
) -> Result<Array2<i32>> {
    if y_true.shape() != y_pred.shape() {
        return Err(MetricsError::InvalidInput(
            "y_true and y_pred must have the same shape".to_string(),
        ));
    }

    let (n_samples, n_labels) = y_true.dim();

    if let Some(weights) = sample_weight {
        if weights.len() != n_samples {
            return Err(MetricsError::InvalidInput(
                "sample_weight length must match number of samples".to_string(),
            ));
        }
    }

    let target_labels: Vec<usize> = if let Some(labels) = labels {
        labels.to_vec()
    } else {
        (0..n_labels).collect()
    };

    let mut confusion_matrices = Array2::zeros((target_labels.len() * 2, 2));

    for (label_idx, &label) in target_labels.iter().enumerate() {
        if label >= n_labels {
            return Err(MetricsError::InvalidInput(format!(
                "Label {} is out of bounds for {} labels",
                label, n_labels
            )));
        }

        let mut tp = 0;
        let mut fp = 0;
        let mut tn = 0;
        let mut fn_count = 0;

        for sample_idx in 0..n_samples {
            let true_val = y_true[[sample_idx, label]];
            let pred_val = y_pred[[sample_idx, label]];

            let weight = if let Some(weights) = sample_weight {
                weights[sample_idx] as i32
            } else {
                1
            };

            match (true_val, pred_val) {
                (1, 1) => tp += weight,
                (0, 1) => fp += weight,
                (0, 0) => tn += weight,
                (1, 0) => fn_count += weight,
                _ => {
                    return Err(MetricsError::InvalidInput(
                        "Labels must be 0 or 1 for multilabel classification".to_string(),
                    ))
                }
            }
        }

        let base_idx = label_idx * 2;
        confusion_matrices[[base_idx, 0]] = tn;
        confusion_matrices[[base_idx, 1]] = fp;
        confusion_matrices[[base_idx + 1, 0]] = fn_count;
        confusion_matrices[[base_idx + 1, 1]] = tp;
    }

    Ok(confusion_matrices)
}

/// Equivalent to sklearn.metrics.cohen_kappa_score  
pub fn cohen_kappa_score_sklearn(
    y1: &Array1<i32>,
    y2: &Array1<i32>,
    labels: Option<&[i32]>,
    weights: Option<&str>,
    sample_weight: Option<&Array1<f64>>,
) -> Result<f64> {
    if y1.len() != y2.len() {
        return Err(MetricsError::InvalidInput(
            "y1 and y2 must have the same length".to_string(),
        ));
    }

    if let Some(sw) = sample_weight {
        if sw.len() != y1.len() {
            return Err(MetricsError::InvalidInput(
                "sample_weight length must match y1 and y2 length".to_string(),
            ));
        }
    }

    // Determine unique labels
    let unique_labels: Vec<i32> = if let Some(labels) = labels {
        labels.to_vec()
    } else {
        let all_labels: HashSet<i32> = y1.iter().chain(y2.iter()).copied().collect();
        let mut sorted_labels: Vec<i32> = all_labels.into_iter().collect();
        sorted_labels.sort();
        sorted_labels
    };

    let n_labels = unique_labels.len();
    let _n = y1.len();

    // Create confusion matrix
    let mut confusion_matrix = Array2::zeros((n_labels, n_labels));
    let mut total_weight = 0.0;

    for (idx, (&true_val, &pred_val)) in y1.iter().zip(y2.iter()).enumerate() {
        let weight = if let Some(sw) = sample_weight {
            sw[idx]
        } else {
            1.0
        };

        if let (Some(true_idx), Some(pred_idx)) = (
            unique_labels.iter().position(|&x| x == true_val),
            unique_labels.iter().position(|&x| x == pred_val),
        ) {
            confusion_matrix[[true_idx, pred_idx]] += weight;
            total_weight += weight;
        }
    }

    // Normalize confusion matrix
    if total_weight > 0.0 {
        confusion_matrix /= total_weight;
    }

    // Calculate observed agreement (diagonal sum)
    let mut po = 0.0;
    for i in 0..n_labels {
        po += confusion_matrix[[i, i]];
    }

    // Calculate expected agreement
    let mut pe = 0.0;
    match weights {
        Some("linear") => {
            // Linear weights: w_ij = 1 - |i - j| / (n_labels - 1)
            for i in 0..n_labels {
                for j in 0..n_labels {
                    let weight_ij = 1.0 - (i as f64 - j as f64).abs() / (n_labels - 1) as f64;
                    let row_sum = confusion_matrix.row(i).sum();
                    let col_sum = confusion_matrix.column(j).sum();
                    pe += weight_ij * row_sum * col_sum;
                }
            }
        }
        Some("quadratic") => {
            // Quadratic weights: w_ij = 1 - ((i - j) / (n_labels - 1))^2
            for i in 0..n_labels {
                for j in 0..n_labels {
                    let diff = (i as f64 - j as f64) / (n_labels - 1) as f64;
                    let weight_ij = 1.0 - diff * diff;
                    let row_sum = confusion_matrix.row(i).sum();
                    let col_sum = confusion_matrix.column(j).sum();
                    pe += weight_ij * row_sum * col_sum;
                }
            }
        }
        None => {
            // Standard Cohen's kappa (no weighting)
            for i in 0..n_labels {
                let row_sum = confusion_matrix.row(i).sum();
                let col_sum = confusion_matrix.column(i).sum();
                pe += row_sum * col_sum;
            }
        }
        _ => {
            return Err(MetricsError::InvalidInput(
                "weights must be None, 'linear', or 'quadratic'".to_string(),
            ))
        }
    }

    // Calculate kappa
    if (1.0 - pe).abs() < 1e-15 {
        Ok(1.0) // Perfect agreement
    } else {
        Ok((po - pe) / (1.0 - pe))
    }
}

/// Equivalent to sklearn.metrics.hinge_loss
pub fn hinge_loss_sklearn(
    y_true: &Array1<i32>,
    y_pred: &Array2<f64>,
    labels: Option<&[i32]>,
    sample_weight: Option<&Array1<f64>>,
) -> Result<f64> {
    let (n_samples, n_classes) = y_pred.dim();

    if y_true.len() != n_samples {
        return Err(MetricsError::InvalidInput(
            "y_true length must match number of samples in y_pred".to_string(),
        ));
    }

    if let Some(sw) = sample_weight {
        if sw.len() != n_samples {
            return Err(MetricsError::InvalidInput(
                "sample_weight length must match number of samples".to_string(),
            ));
        }
    }

    // Determine class labels
    let class_labels: Vec<i32> = if let Some(labels) = labels {
        if labels.len() != n_classes {
            return Err(MetricsError::InvalidInput(
                "labels length must match number of classes in y_pred".to_string(),
            ));
        }
        labels.to_vec()
    } else {
        let unique_labels: HashSet<i32> = y_true.iter().copied().collect();
        let mut sorted_labels: Vec<i32> = unique_labels.into_iter().collect();
        sorted_labels.sort();
        if sorted_labels.len() != n_classes {
            return Err(MetricsError::InvalidInput(
                "Number of unique labels in y_true must match number of classes in y_pred"
                    .to_string(),
            ));
        }
        sorted_labels
    };

    let mut total_loss = 0.0;
    let mut total_weight = 0.0;

    for (sample_idx, &true_label) in y_true.iter().enumerate() {
        let weight = if let Some(sw) = sample_weight {
            sw[sample_idx]
        } else {
            1.0
        };

        // Find the index of the true label
        if let Some(true_class_idx) = class_labels.iter().position(|&x| x == true_label) {
            let true_score = y_pred[[sample_idx, true_class_idx]];

            // Calculate hinge loss for this sample
            let mut sample_loss = 0.0;
            for (class_idx, &_class_label) in class_labels.iter().enumerate() {
                if class_idx != true_class_idx {
                    let class_score = y_pred[[sample_idx, class_idx]];
                    let margin = true_score - class_score;
                    sample_loss += (1.0 - margin).max(0.0);
                }
            }

            total_loss += weight * sample_loss;
            total_weight += weight;
        } else {
            return Err(MetricsError::InvalidInput(format!(
                "Label {} not found in provided labels",
                true_label
            )));
        }
    }

    if total_weight > 0.0 {
        Ok(total_loss / total_weight)
    } else {
        Ok(0.0)
    }
}

/// Equivalent to sklearn.metrics.zero_one_loss
pub fn zero_one_loss_sklearn(
    y_true: &Array1<i32>,
    y_pred: &Array1<i32>,
    normalize: bool,
    sample_weight: Option<&Array1<f64>>,
) -> Result<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::InvalidInput(
            "y_true and y_pred must have the same length".to_string(),
        ));
    }

    if let Some(sw) = sample_weight {
        if sw.len() != y_true.len() {
            return Err(MetricsError::InvalidInput(
                "sample_weight length must match y_true and y_pred length".to_string(),
            ));
        }
    }

    let mut total_errors = 0.0;
    let mut total_weight = 0.0;

    for (idx, (&true_val, &pred_val)) in y_true.iter().zip(y_pred.iter()).enumerate() {
        let weight = if let Some(sw) = sample_weight {
            sw[idx]
        } else {
            1.0
        };

        if true_val != pred_val {
            total_errors += weight;
        }
        total_weight += weight;
    }

    if normalize {
        if total_weight > 0.0 {
            Ok(total_errors / total_weight)
        } else {
            Ok(0.0)
        }
    } else {
        Ok(total_errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_classification_report_sklearn() {
        let y_true = Array1::from_vec(vec![0, 1, 2, 0, 1, 2]);
        let y_pred = Array1::from_vec(vec![0, 2, 1, 0, 0, 2]);

        let report = classification_report_sklearn(&y_true, &y_pred, None, None, 2, 0.0).unwrap();

        assert!(report.accuracy >= 0.0 && report.accuracy <= 1.0);
        assert!(report.precision.len() == 3);
        assert!(report.recall.len() == 3);
        assert!(report.f1_score.len() == 3);
    }

    #[test]
    fn test_precision_recall_fscore_support_sklearn() {
        let y_true = Array1::from_vec(vec![0, 1, 2, 0, 1, 2]);
        let y_pred = Array1::from_vec(vec![0, 2, 1, 0, 0, 2]);

        let (precision, recall, fscore, support) = precision_recall_fscore_support_sklearn(
            &y_true,
            &y_pred,
            1.0,
            None,
            None,
            Some("macro"),
            None,
            0.0,
        )
        .unwrap();

        assert_eq!(precision.len(), 1);
        assert_eq!(recall.len(), 1);
        assert_eq!(fscore.len(), 1);
        assert_eq!(support.len(), 1);
    }

    #[test]
    fn test_cohen_kappa_score_sklearn() {
        let y1 = Array1::from_vec(vec![0, 1, 0, 1]);
        let y2 = Array1::from_vec(vec![0, 1, 0, 1]);

        let kappa = cohen_kappa_score_sklearn(&y1, &y2, None, None, None).unwrap();
        assert!((kappa - 1.0).abs() < 1e-10); // Perfect agreement

        let y3 = Array1::from_vec(vec![0, 1, 1, 0]);
        let kappa2 = cohen_kappa_score_sklearn(&y1, &y3, None, None, None).unwrap();
        assert!(kappa2 < 1.0); // Less than perfect agreement
    }

    #[test]
    fn test_zero_one_loss_sklearn() {
        let y_true = Array1::from_vec(vec![0, 1, 0, 1]);
        let y_pred = Array1::from_vec(vec![0, 1, 1, 0]);

        let loss_normalized = zero_one_loss_sklearn(&y_true, &y_pred, true, None).unwrap();
        assert!((loss_normalized - 0.5).abs() < 1e-10); // 2 errors out of 4

        let loss_count = zero_one_loss_sklearn(&y_true, &y_pred, false, None).unwrap();
        assert!((loss_count - 2.0).abs() < 1e-10); // 2 errors
    }

    #[test]
    fn test_multilabel_confusion_matrix_sklearn() {
        let y_true =
            Array2::from_shape_vec((4, 3), vec![1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1]).unwrap();

        let y_pred =
            Array2::from_shape_vec((4, 3), vec![1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1]).unwrap();

        let confusion_matrices =
            multilabel_confusion_matrix_sklearn(&y_true, &y_pred, None, None).unwrap();

        assert_eq!(confusion_matrices.shape(), [6, 2]); // 3 labels * 2 rows each
    }
}
