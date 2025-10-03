//! Fairness and Bias Metrics module
//!
//! This module provides metrics for evaluating fairness and bias in machine learning models.
//! These metrics help identify and quantify disparities in model performance or predictions
//! across different protected groups.
//!
//! ## Group Fairness Metrics
//!
//! Group fairness metrics measure whether a model treats different demographic groups equally.
//! The module includes several common fairness criteria, such as Demographic Parity, Equalized Odds,
//! and Equal Opportunity.
//!
//! ```
//! use scirs2_core::ndarray::array;
//! use scirs2_metrics::fairness::{
//!     demographic_parity_difference, equalized_odds_difference, equal_opportunity_difference
//! };
//!
//! // Example: binary predictions for two protected groups
//! // y_true: ground truth labels (0 or 1)
//! let y_true = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
//! // y_pred: predicted labels (0 or 1)
//! let y_pred = array![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0];
//! // is_protected: boolean array indicating protected group membership (1 for protected group, 0 otherwise)
//! let protected_group = array![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
//!
//! // Compute demographic parity difference
//! // A value of 0 indicates perfect demographic parity
//! let dp_diff = demographic_parity_difference(&y_pred, &protected_group).unwrap();
//!
//! // Compute equalized odds difference
//! // A value of 0 indicates that the false positive and true positive rates are
//! // the same for both groups
//! let eod_diff = equalized_odds_difference(&y_true, &y_pred, &protected_group).unwrap();
//!
//! // Compute equal opportunity difference
//! // A value of 0 indicates equal true positive rates across groups
//! let eo_diff = equal_opportunity_difference(&y_true, &y_pred, &protected_group).unwrap();
//!
//! // Calculate disparate impact
//! // A value of 1.0 indicates perfect fairness; less than 0.8 or greater than 1.25
//! // is often considered problematic
//! let di = scirs2_metrics::fairness::disparate_impact(&y_pred, &protected_group).unwrap();
//! ```
//!
//! ## Consistency Measures
//!
//! Consistency measures evaluate whether similar individuals receive similar predictions,
//! regardless of protected attributes.
//!
//! ```
//! use scirs2_core::ndarray::{array, Array2};
//! use scirs2_metrics::fairness::consistency_score;
//!
//! // Features matrix: each row is an individual, each column is a feature
//! let features = Array2::from_shape_vec((6, 2),
//!     vec![0.1, 0.2, 0.15, 0.21, 0.9, 0.8, 0.92, 0.79, 0.5, 0.51, 0.52, 0.49]
//! ).unwrap();
//!
//! // Predictions for each individual
//! let predictions = array![0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
//!
//! // Calculate consistency score (higher is better)
//! let consistency = consistency_score(&features, &predictions, 2).unwrap();
//! ```
//!
//! ## Comprehensive Bias Detection
//!
//! The `bias_detection` submodule provides tools for more in-depth analysis of fairness
//! and bias in machine learning models, including:
//!
//! - Slicing analysis utilities to evaluate model performance across different data slices
//! - Subgroup performance metrics to identify disparities across demographic subgroups
//! - Intersectional fairness measures to analyze bias at the intersection of multiple protected attributes
//!
//! ## Robustness Metrics
//!
//! The `robustness` submodule provides tools to evaluate the stability and reliability of
//! fairness metrics under different conditions:
//!
//! - Performance invariance measures to evaluate consistency across different demographic groups
//! - Influence functions to identify which samples most impact fairness metrics
//! - Sensitivity to perturbations to assess how model fairness changes with data modifications
//!
//! ```
//! use scirs2_core::ndarray::{array, Array2};
//! use scirs2_metrics::fairness::{demographic_parity_difference, disparate_impact};
//! use scirs2_metrics::fairness::robustness::{
//!     performance_invariance, influence_function, perturbation_sensitivity, PerturbationType
//! };
//!
//! // Sample data
//! let y_true = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
//! let y_pred = array![0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0];
//! let protected_group = array![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
//!
//! // Measure influence of individual samples on fairness
//! let influence_scores = influence_function(
//!     &y_true,
//!     &y_pred,
//!     &protected_group,
//!     |yp, pg| demographic_parity_difference(yp, pg).unwrap_or(1.0),
//!     None
//! ).unwrap();
//!
//! // Evaluate sensitivity to perturbations
//! let sensitivity = perturbation_sensitivity(
//!     &y_true,
//!     &y_pred,
//!     &protected_group,
//!     PerturbationType::LabelFlip,
//!     |yp, pg| demographic_parity_difference(yp, pg).unwrap_or(1.0),
//!     0.1,  // 10% perturbation level
//!     5,    // 5 iterations
//!     Some(42)  // Random seed
//! ).unwrap();
//! ```
//!
//! ```
//! use scirs2_core::ndarray::{array, Array2};
//! use scirs2_metrics::fairness::bias_detection::{
//!     slice_analysis, subgroup_performance, intersectional_fairness
//! };
//! use scirs2_metrics::classification::accuracy_score;
//!
//! // For demo purposes, create a simple dataset
//! let y_true = array![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
//! let y_pred = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0];
//!
//! // Protected attributes: gender (0=male, 1=female) and race (0=group A, 1=group B)
//! let protected_features = Array2::from_shape_vec((8, 2), vec![
//!     0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
//! ]).unwrap();
//!
//! let feature_names = vec!["gender".to_string(), "race".to_string()];
//!
//! // Analyze intersectional fairness
//! let fairness_results = intersectional_fairness(
//!     &y_true,
//!     &y_pred,
//!     &protected_features,
//!     &feature_names
//! ).unwrap();
//!
//! // Evaluate performance across different subgroups
//! let performance_results = subgroup_performance(
//!     &y_true,
//!     &y_pred,
//!     &protected_features,
//!     &feature_names,
//!     |y_t, y_p| {
//!         // Convert Vec<f64> to Array1<f64> for accuracy_score
//!         let y_t_array = scirs2_core::ndarray::Array::from_vec(y_t.to_vec());
//!         let y_p_array = scirs2_core::ndarray::Array::from_vec(y_p.to_vec());
//!         accuracy_score(&y_t_array, &y_p_array).unwrap_or(0.0)
//!     }
//! ).unwrap();
//! ```

use scirs2_core::ndarray::{ArrayBase, Data, Ix1, Ix2};
use scirs2_core::numeric::Float;
use std::cmp::Ordering;

use crate::error::{MetricsError, Result};

// Expose the submodules
pub mod bias_detection;
pub mod robustness;

/// Calculates the demographic parity difference between groups.
///
/// Demographic parity (also known as statistical parity) is satisfied when the proportion
/// of positive predictions is equal across all protected attribute groups.
/// The difference is calculated as:
/// |Probability(prediction=1 | group=0) - Probability(prediction=1 | group=1)|
///
/// # Arguments
///
/// * `y_pred` - Predicted labels (binary: 0 or 1)
/// * `protected_group` - Binary indicator of protected group membership (1 for protected, 0 otherwise)
///
/// # Returns
///
/// * The absolute difference in positive prediction rates between groups (between 0 and 1)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_metrics::fairness::demographic_parity_difference;
///
/// let y_pred = array![1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0];
/// let protected_group = array![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
///
/// // Calculate the difference in positive prediction rates
/// let dp_diff = demographic_parity_difference(&y_pred, &protected_group).unwrap();
///
/// // Interpretation:
/// // - 0 indicates perfect demographic parity (equal prediction rates)
/// // - Values further from 0 indicate greater disparity
/// ```
#[allow(dead_code)]
pub fn demographic_parity_difference<T, S, R>(
    y_pred: &ArrayBase<S, Ix1>,
    protected_group: &ArrayBase<R, Ix1>,
) -> Result<f64>
where
    T: Float + PartialOrd + Clone,
    S: Data<Elem = T>,
    R: Data<Elem = T>,
{
    // Check that arrays have the same length
    if y_pred.len() != protected_group.len() {
        return Err(MetricsError::InvalidInput(format!(
            "Arrays have different lengths: {} vs {}",
            y_pred.len(),
            protected_group.len()
        )));
    }

    if y_pred.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    let zero = T::zero();

    // Count positive predictions in each group
    let mut protected_group_positive = 0;
    let mut protected_group_total = 0;
    let mut unprotected_group_positive = 0;
    let mut unprotected_group_total = 0;

    for (pred, group) in y_pred.iter().zip(protected_group.iter()) {
        if group > &zero {
            // Protected group
            protected_group_total += 1;
            if pred > &zero {
                protected_group_positive += 1;
            }
        } else {
            // Unprotected group
            unprotected_group_total += 1;
            if pred > &zero {
                unprotected_group_positive += 1;
            }
        }
    }

    // Check if there are members in both groups
    if protected_group_total == 0 || unprotected_group_total == 0 {
        return Err(MetricsError::InvalidInput(
            "Each group must have at least one member".to_string(),
        ));
    }

    // Calculate positive prediction rates for each group
    let protected_rate = protected_group_positive as f64 / protected_group_total as f64;
    let unprotected_rate = unprotected_group_positive as f64 / unprotected_group_total as f64;

    // Return the absolute difference
    Ok((protected_rate - unprotected_rate).abs())
}

/// Calculates the disparate impact ratio between groups.
///
/// Disparate impact measures the ratio of positive prediction rates between groups.
/// A ratio close to 1.0 indicates fairness, while values far from 1.0 suggest bias.
/// The formula is:
/// Probability(prediction=1 | group=1) / Probability(prediction=1 | group=0)
///
/// # Arguments
///
/// * `y_pred` - Predicted labels (binary: 0 or 1)
/// * `protected_group` - Binary indicator of protected group membership (1 for protected, 0 otherwise)
///
/// # Returns
///
/// * The ratio of positive prediction rates (protected group rate / unprotected group rate)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_metrics::fairness::disparate_impact;
///
/// let y_pred = array![1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0];
/// let protected_group = array![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
///
/// // Calculate the disparate impact ratio
/// let di = disparate_impact(&y_pred, &protected_group).unwrap();
///
/// // Interpretation:
/// // - 1.0 indicates perfect fairness (equal prediction rates)
/// // - < 0.8 or > 1.25 may indicate problematic disparate impact
/// // - In the US, the 80% rule (di < 0.8) is often used as a legal threshold
/// ```
#[allow(dead_code)]
pub fn disparate_impact<T, S, R>(
    y_pred: &ArrayBase<S, Ix1>,
    protected_group: &ArrayBase<R, Ix1>,
) -> Result<f64>
where
    T: Float + PartialOrd + Clone,
    S: Data<Elem = T>,
    R: Data<Elem = T>,
{
    // Check that arrays have the same length
    if y_pred.len() != protected_group.len() {
        return Err(MetricsError::InvalidInput(format!(
            "Arrays have different lengths: {} vs {}",
            y_pred.len(),
            protected_group.len()
        )));
    }

    if y_pred.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    let zero = T::zero();

    // Count positive predictions in each group
    let mut protected_group_positive = 0;
    let mut protected_group_total = 0;
    let mut unprotected_group_positive = 0;
    let mut unprotected_group_total = 0;

    for (pred, group) in y_pred.iter().zip(protected_group.iter()) {
        if group > &zero {
            // Protected group
            protected_group_total += 1;
            if pred > &zero {
                protected_group_positive += 1;
            }
        } else {
            // Unprotected group
            unprotected_group_total += 1;
            if pred > &zero {
                unprotected_group_positive += 1;
            }
        }
    }

    // Check if there are members in both groups
    if protected_group_total == 0 || unprotected_group_total == 0 {
        return Err(MetricsError::InvalidInput(
            "Each group must have at least one member".to_string(),
        ));
    }

    // Calculate positive prediction rates for each group
    let protected_rate = protected_group_positive as f64 / protected_group_total as f64;
    let unprotected_rate = unprotected_group_positive as f64 / unprotected_group_total as f64;

    // Prevent division by zero
    if unprotected_rate == 0.0 {
        if protected_rate == 0.0 {
            // Both rates are zero, so perfectly equal
            return Ok(1.0);
        }
        return Ok(f64::INFINITY);
    }

    // Return the ratio
    Ok(protected_rate / unprotected_rate)
}

/// Calculates the equalized odds difference between groups.
///
/// Equalized odds is satisfied when both the false positive rate and true positive rate
/// are equal across all protected attribute groups. This function calculates the maximum
/// absolute difference in these rates between the protected and unprotected groups.
///
/// # Arguments
///
/// * `y_true` - Ground truth labels (binary: 0 or 1)
/// * `y_pred` - Predicted labels (binary: 0 or 1)
/// * `protected_group` - Binary indicator of protected group membership (1 for protected, 0 otherwise)
///
/// # Returns
///
/// * The maximum absolute difference in false positive and true positive rates (between 0 and 1)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_metrics::fairness::equalized_odds_difference;
///
/// let y_true = array![0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0];
/// let y_pred = array![1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0];
/// let protected_group = array![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
///
/// // Calculate the equalized odds difference
/// let eod_diff = equalized_odds_difference(&y_true, &y_pred, &protected_group).unwrap();
///
/// // Interpretation:
/// // - 0 indicates perfect equalized odds
/// // - Higher values indicate greater disparity in error rates
/// ```
#[allow(dead_code)]
pub fn equalized_odds_difference<T, S, R, Q>(
    y_true: &ArrayBase<S, Ix1>,
    y_pred: &ArrayBase<R, Ix1>,
    protected_group: &ArrayBase<Q, Ix1>,
) -> Result<f64>
where
    T: Float + PartialOrd + Clone,
    S: Data<Elem = T>,
    R: Data<Elem = T>,
    Q: Data<Elem = T>,
{
    // Check that all arrays have the same length
    if y_true.len() != y_pred.len() || y_true.len() != protected_group.len() {
        return Err(MetricsError::InvalidInput(format!(
            "Arrays have different lengths: {}, {}, {}",
            y_true.len(),
            y_pred.len(),
            protected_group.len()
        )));
    }

    if y_true.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    let zero = T::zero();

    // Initialize counters for each group and outcome
    let mut protected_true_positives = 0;
    let mut protected_false_positives = 0;
    let mut protected_true_negatives = 0;
    let mut protected_false_negatives = 0;

    let mut unprotected_true_positives = 0;
    let mut unprotected_false_positives = 0;
    let mut unprotected_true_negatives = 0;
    let mut unprotected_false_negatives = 0;

    // Calculate confusion matrix values for each group
    for ((truth, pred), group) in y_true.iter().zip(y_pred.iter()).zip(protected_group.iter()) {
        if group > &zero {
            // Protected group
            if truth > &zero {
                // Positive class
                if pred > &zero {
                    protected_true_positives += 1;
                } else {
                    protected_false_negatives += 1;
                }
            } else {
                // Negative class
                if pred > &zero {
                    protected_false_positives += 1;
                } else {
                    protected_true_negatives += 1;
                }
            }
        } else {
            // Unprotected group
            if truth > &zero {
                // Positive class
                if pred > &zero {
                    unprotected_true_positives += 1;
                } else {
                    unprotected_false_negatives += 1;
                }
            } else {
                // Negative class
                if pred > &zero {
                    unprotected_false_positives += 1;
                } else {
                    unprotected_true_negatives += 1;
                }
            }
        }
    }

    // Calculate _true positive rates (TPR) and false positive rates (FPR) for each group
    // Handle cases where a group might not have any positives or negatives
    let protected_tpr = if protected_true_positives + protected_false_negatives > 0 {
        protected_true_positives as f64
            / (protected_true_positives + protected_false_negatives) as f64
    } else {
        0.0
    };

    let protected_fpr = if protected_false_positives + protected_true_negatives > 0 {
        protected_false_positives as f64
            / (protected_false_positives + protected_true_negatives) as f64
    } else {
        0.0
    };

    let unprotected_tpr = if unprotected_true_positives + unprotected_false_negatives > 0 {
        unprotected_true_positives as f64
            / (unprotected_true_positives + unprotected_false_negatives) as f64
    } else {
        0.0
    };

    let unprotected_fpr = if unprotected_false_positives + unprotected_true_negatives > 0 {
        unprotected_false_positives as f64
            / (unprotected_false_positives + unprotected_true_negatives) as f64
    } else {
        0.0
    };

    // Calculate the absolute differences in rates
    let tpr_diff = (protected_tpr - unprotected_tpr).abs();
    let fpr_diff = (protected_fpr - unprotected_fpr).abs();

    // Return the maximum of the two differences
    Ok(tpr_diff.max(fpr_diff))
}

/// Calculates the equal opportunity difference between groups.
///
/// Equal opportunity is satisfied when the true positive rates are equal across all
/// protected attribute groups. This is a relaxed version of equalized odds that only
/// considers the true positive rate.
///
/// # Arguments
///
/// * `y_true` - Ground truth labels (binary: 0 or 1)
/// * `y_pred` - Predicted labels (binary: 0 or 1)
/// * `protected_group` - Binary indicator of protected group membership (1 for protected, 0 otherwise)
///
/// # Returns
///
/// * The absolute difference in true positive rates between groups (between 0 and 1)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_metrics::fairness::equal_opportunity_difference;
///
/// let y_true = array![0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0];
/// let y_pred = array![1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0];
/// let protected_group = array![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
///
/// // Calculate the equal opportunity difference
/// let eo_diff = equal_opportunity_difference(&y_true, &y_pred, &protected_group).unwrap();
///
/// // Interpretation:
/// // - 0 indicates perfect equal opportunity
/// // - Higher values indicate greater disparity in true positive rates
/// ```
#[allow(dead_code)]
pub fn equal_opportunity_difference<T, S, R, Q>(
    y_true: &ArrayBase<S, Ix1>,
    y_pred: &ArrayBase<R, Ix1>,
    protected_group: &ArrayBase<Q, Ix1>,
) -> Result<f64>
where
    T: Float + PartialOrd + Clone,
    S: Data<Elem = T>,
    R: Data<Elem = T>,
    Q: Data<Elem = T>,
{
    // Check that all arrays have the same length
    if y_true.len() != y_pred.len() || y_true.len() != protected_group.len() {
        return Err(MetricsError::InvalidInput(format!(
            "Arrays have different lengths: {}, {}, {}",
            y_true.len(),
            y_pred.len(),
            protected_group.len()
        )));
    }

    if y_true.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    let zero = T::zero();

    // Initialize counters for _true positives and false negatives in each group
    let mut protected_true_positives = 0;
    let mut protected_false_negatives = 0;
    let mut unprotected_true_positives = 0;
    let mut unprotected_false_negatives = 0;

    // Count _true positives and false negatives for each group
    for ((truth, pred), group) in y_true.iter().zip(y_pred.iter()).zip(protected_group.iter()) {
        if truth > &zero {
            // Only consider cases where _true label is positive
            if group > &zero {
                // Protected group
                if pred > &zero {
                    protected_true_positives += 1;
                } else {
                    protected_false_negatives += 1;
                }
            } else {
                // Unprotected group
                if pred > &zero {
                    unprotected_true_positives += 1;
                } else {
                    unprotected_false_negatives += 1;
                }
            }
        }
    }

    // Calculate _true positive rates for each group
    let protected_tpr = if protected_true_positives + protected_false_negatives > 0 {
        protected_true_positives as f64
            / (protected_true_positives + protected_false_negatives) as f64
    } else {
        // If there are no positive examples in the protected group
        return Err(MetricsError::InvalidInput(
            "No positive examples in protected group".to_string(),
        ));
    };

    let unprotected_tpr = if unprotected_true_positives + unprotected_false_negatives > 0 {
        unprotected_true_positives as f64
            / (unprotected_true_positives + unprotected_false_negatives) as f64
    } else {
        // If there are no positive examples in the unprotected group
        return Err(MetricsError::InvalidInput(
            "No positive examples in unprotected group".to_string(),
        ));
    };

    // Return the absolute difference in _true positive rates
    Ok((protected_tpr - unprotected_tpr).abs())
}

/// Calculates the consistency score of a model's predictions.
///
/// Consistency measures whether similar individuals receive similar predictions,
/// regardless of protected attributes. For each instance, it finds the k nearest
/// neighbors and calculates the average absolute difference between the instance's
/// prediction and the predictions of its neighbors.
///
/// # Arguments
///
/// * `features` - Feature matrix (each row is an instance, each column is a feature)
/// * `predictions` - Predicted labels or scores for each instance
/// * `k` - Number of nearest neighbors to consider for each instance
///
/// # Returns
///
/// * The consistency score (between 0 and 1, higher is better)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, Array2};
/// use scirs2_metrics::fairness::consistency_score;
///
/// // Features matrix (6 instances, 2 features each)
/// let features = Array2::from_shape_vec((6, 2),
///     vec![0.1, 0.2, 0.15, 0.21, 0.9, 0.8, 0.92, 0.79, 0.5, 0.51, 0.52, 0.49]
/// ).unwrap();
///
/// // Predictions for each instance
/// let predictions = array![0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
///
/// // Calculate consistency with 2 neighbors
/// let consistency = consistency_score(&features, &predictions, 2).unwrap();
/// ```
#[allow(dead_code)]
pub fn consistency_score<T, S, R>(
    features: &ArrayBase<S, Ix2>,
    predictions: &ArrayBase<R, Ix1>,
    k: usize,
) -> Result<f64>
where
    T: Float + PartialOrd + Clone,
    S: Data<Elem = T>,
    R: Data<Elem = T>,
{
    // Check dimensions
    let n_samples = features.nrows();
    if n_samples != predictions.len() {
        return Err(MetricsError::InvalidInput(format!(
            "Number of samples in features ({}) and predictions ({}) do not match",
            n_samples,
            predictions.len()
        )));
    }

    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    // Check if k is valid (less than n_samples)
    if k >= n_samples {
        return Err(MetricsError::InvalidInput(format!(
            "k ({}) must be less than the number of samples ({})",
            k, n_samples
        )));
    }

    if k == 0 {
        return Err(MetricsError::InvalidInput(
            "k must be greater than 0".to_string(),
        ));
    }

    // Calculate pairwise distances between all instances
    let mut distances = Vec::with_capacity(n_samples * n_samples);
    for i in 0..n_samples {
        for j in 0..n_samples {
            if i != j {
                // Calculate Euclidean distance between instances i and j
                let mut dist = 0.0;
                for c in 0..features.ncols() {
                    let diff = features[[i, c]].to_f64().unwrap_or(0.0)
                        - features[[j, c]].to_f64().unwrap_or(0.0);
                    dist += diff * diff;
                }
                dist = dist.sqrt();
                distances.push((i, j, dist));
            }
        }
    }

    // For each instance, find k nearest neighbors
    let mut consistency_sum = 0.0;
    for i in 0..n_samples {
        // Get distances from instance i to all other instances
        let mut neighbors: Vec<_> = distances.iter().filter(|(idx, _, _)| *idx == i).collect();

        // Sort by distance
        neighbors.sort_by(|(_, _, dist_a), (_, _, dist_b)| {
            dist_a.partial_cmp(dist_b).unwrap_or(Ordering::Equal)
        });

        // Get predictions of k nearest neighbors
        let nearest_k = neighbors
            .iter()
            .take(k)
            .map(|(_, j, _)| *j)
            .collect::<Vec<_>>();

        // Calculate mean absolute difference between prediction and neighbors' predictions
        let pred_i = predictions[i].to_f64().unwrap_or(0.0);
        let mut diff_sum = 0.0;

        for &j in &nearest_k {
            let pred_j = predictions[j].to_f64().unwrap_or(0.0);
            diff_sum += (pred_i - pred_j).abs();
        }

        let mean_diff = diff_sum / k as f64;
        consistency_sum += mean_diff;
    }

    // Return 1 - average inconsistency (so higher is better)
    // Scale to [0, 1] assuming predictions are binary (0/1)
    Ok(1.0 - consistency_sum / n_samples as f64)
}
