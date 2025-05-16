//! Comprehensive bias detection utilities for fairness assessment
//!
//! This module provides tools to detect and analyze bias in machine learning models
//! across different slices of data, evaluate subgroup performance, and measure
//! intersectional fairness.

use ndarray::{ArrayBase, Data, Ix1, Ix2};
use num_traits::real::Real;
use std::collections::{BTreeSet, HashMap};

use crate::error::{MetricsError, Result};

/// Represents a subset of data defined by specific feature conditions
#[derive(Debug, Clone)]
pub struct DataSlice {
    /// Name of the slice
    pub name: String,
    /// Mask indicating which instances belong to this slice
    pub mask: Vec<bool>,
    /// Description of the slicing criteria
    pub description: Option<String>,
}

/// Performs slicing analysis on a dataset based on feature values.
///
/// This function creates slices of data based on unique values in categorical features,
/// and evaluates model performance on each slice to identify potential bias.
///
/// # Arguments
///
/// * `features` - Matrix of features, where each row is an instance and each column is a feature
/// * `categorical_features` - Indices of categorical features to use for slicing
/// * `y_true` - Ground truth labels
/// * `y_pred` - Predicted labels or scores
/// * `metric_fn` - Closure that computes a performance metric given true and predicted labels
///
/// # Returns
///
/// * HashMap mapping slice names to their performance metric values
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array2};
/// use scirs2_metrics::fairness::bias_detection::slice_analysis;
/// use scirs2_metrics::classification::accuracy_score;
///
/// // Create sample dataset
/// let features = Array2::from_shape_vec((8, 3), vec![
///     // age, gender(0=male, 1=female), region(0,1,2)
///     25.0, 0.0, 0.0,
///     30.0, 0.0, 1.0,
///     22.0, 1.0, 0.0,
///     35.0, 1.0, 1.0,
///     40.0, 0.0, 2.0,
///     45.0, 0.0, 0.0,
///     28.0, 1.0, 2.0,
///     50.0, 1.0, 0.0,
/// ]).unwrap();
///
/// let y_true = array![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
/// let y_pred = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0];
///
/// // Analyze slices based on gender (column 1) and region (column 2)
/// let results = slice_analysis(
///     &features,
///     &[1, 2],
///     &y_true,
///     &y_pred,
///     |y_t, y_p| {
///         // Convert Vec<f64> to Array1<f64> for accuracy_score
///         let y_t_array = ndarray::Array::from_vec(y_t.to_vec());
///         let y_p_array = ndarray::Array::from_vec(y_p.to_vec());
///         accuracy_score(&y_t_array, &y_p_array).unwrap_or(0.0)
///     }
/// ).unwrap();
///
/// // Examine performance across different slices
/// for (slice_name, metric_value) in results {
///     println!("Slice: {}, Accuracy: {:.2}", slice_name, metric_value);
/// }
/// ```
pub fn slice_analysis<T, S1, S2, S3, F>(
    features: &ArrayBase<S1, Ix2>,
    categorical_features: &[usize],
    y_true: &ArrayBase<S2, Ix1>,
    y_pred: &ArrayBase<S3, Ix1>,
    metric_fn: F,
) -> Result<HashMap<String, f64>>
where
    T: Real + PartialOrd + Clone,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    S3: Data<Elem = T>,
    F: Fn(&[T], &[T]) -> f64,
{
    // Check that dimensions match
    let n_samples = features.nrows();
    if n_samples != y_true.len() || n_samples != y_pred.len() {
        return Err(MetricsError::InvalidInput(format!(
            "Dimensions mismatch: features ({} rows), y_true ({}), y_pred ({})",
            n_samples,
            y_true.len(),
            y_pred.len()
        )));
    }

    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    let n_features = features.ncols();
    for &feature_idx in categorical_features {
        if feature_idx >= n_features {
            return Err(MetricsError::InvalidInput(format!(
                "Feature index {} out of bounds (max: {})",
                feature_idx,
                n_features - 1
            )));
        }
    }

    let mut results = HashMap::new();

    // Get overall metric for the entire dataset
    let y_true_vec: Vec<T> = y_true.iter().cloned().collect();
    let y_pred_vec: Vec<T> = y_pred.iter().cloned().collect();
    results.insert("overall".to_string(), metric_fn(&y_true_vec, &y_pred_vec));

    // Create slices for each categorical feature and its values
    for &feature_idx in categorical_features {
        // Find unique values for this feature
        let mut unique_values = BTreeSet::new();
        for i in 0..n_samples {
            let value = features[[i, feature_idx]].to_f64().unwrap();
            // Convert to integer by rounding to handle precision issues
            let rounded_value = (value * 1000.0).round() as i64;
            unique_values.insert(rounded_value);
        }

        // For each unique value, create a slice
        for int_value in unique_values {
            // Convert back to f64 for display and comparison
            let value = int_value as f64 / 1000.0;
            let slice_name = format!("feature_{}_{}", feature_idx, value);

            // Create mask for this slice
            let mut mask = vec![false; n_samples];
            let mut slice_y_true = Vec::new();
            let mut slice_y_pred = Vec::new();

            for i in 0..n_samples {
                let feature_value = features[[i, feature_idx]].to_f64().unwrap();
                let rounded_value = (feature_value * 1000.0).round() as i64;

                if rounded_value == int_value {
                    mask[i] = true;
                    slice_y_true.push(y_true[i].clone());
                    slice_y_pred.push(y_pred[i].clone());
                }
            }

            // Compute metric for this slice
            if !slice_y_true.is_empty() {
                let slice_metric = metric_fn(&slice_y_true, &slice_y_pred);
                results.insert(slice_name, slice_metric);
            }
        }
    }

    Ok(results)
}

/// Calculates performance of a metric across multiple subgroups.
///
/// This function evaluates model performance for different demographic subgroups,
/// allowing for intersectional analysis (e.g., combinations of gender, race, age groups).
///
/// # Arguments
///
/// * `y_true` - Ground truth labels
/// * `y_pred` - Predicted labels or scores
/// * `groups` - Array where each row represents demographic attributes for an instance
/// * `group_names` - Names of the demographic groups (column names in groups)
/// * `metric_fn` - Closure that computes a performance metric given true and predicted labels
///
/// # Returns
///
/// * HashMap mapping subgroup descriptions to their performance metric values
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array2};
/// use scirs2_metrics::fairness::bias_detection::subgroup_performance;
/// use scirs2_metrics::classification::accuracy_score;
///
/// // Create sample dataset
/// let y_true = array![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
/// let y_pred = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0];
///
/// // Demographic groups: gender (0=male, 1=female) and age_group (0=young, 1=old)
/// let groups = Array2::from_shape_vec((8, 2), vec![
///     0.0, 0.0, // male, young
///     0.0, 1.0, // male, old
///     1.0, 0.0, // female, young
///     1.0, 0.0, // female, young
///     0.0, 1.0, // male, old
///     0.0, 1.0, // male, old
///     1.0, 1.0, // female, old
///     1.0, 1.0, // female, old
/// ]).unwrap();
///
/// let group_names = vec!["gender".to_string(), "age_group".to_string()];
///
/// // Analyze performance across subgroups
/// let results = subgroup_performance(
///     &y_true,
///     &y_pred,
///     &groups,
///     &group_names,
///     |y_t, y_p| {
///         // Convert Vec<f64> to Array1<f64> for accuracy_score
///         let y_t_array = ndarray::Array::from_vec(y_t.to_vec());
///         let y_p_array = ndarray::Array::from_vec(y_p.to_vec());
///         accuracy_score(&y_t_array, &y_p_array).unwrap_or(0.0)
///     }
/// ).unwrap();
///
/// // Examine performance across different subgroups
/// for (subgroup, metric_value) in results {
///     println!("Subgroup: {}, Accuracy: {:.2}", subgroup, metric_value);
/// }
/// ```
pub fn subgroup_performance<T, S1, S2, S3, F>(
    y_true: &ArrayBase<S1, Ix1>,
    y_pred: &ArrayBase<S2, Ix1>,
    groups: &ArrayBase<S3, Ix2>,
    group_names: &[String],
    metric_fn: F,
) -> Result<HashMap<String, f64>>
where
    T: Real + PartialOrd + Clone,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    S3: Data<Elem = T>,
    F: Fn(&[T], &[T]) -> f64,
{
    // Check dimensions
    let n_samples = y_true.len();
    if n_samples != y_pred.len() || n_samples != groups.nrows() {
        return Err(MetricsError::InvalidInput(format!(
            "Dimensions mismatch: y_true ({}), y_pred ({}), groups ({} rows)",
            n_samples,
            y_pred.len(),
            groups.nrows()
        )));
    }

    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    let n_groups = groups.ncols();
    if n_groups != group_names.len() {
        return Err(MetricsError::InvalidInput(format!(
            "Number of group columns ({}) doesn't match number of group names ({})",
            n_groups,
            group_names.len()
        )));
    }

    let mut results = HashMap::new();

    // Get overall metric for the entire dataset
    let y_true_vec: Vec<T> = y_true.iter().cloned().collect();
    let y_pred_vec: Vec<T> = y_pred.iter().cloned().collect();
    results.insert("overall".to_string(), metric_fn(&y_true_vec, &y_pred_vec));

    // Find unique values for each group
    let mut unique_values = vec![BTreeSet::new(); n_groups];
    for i in 0..n_samples {
        for j in 0..n_groups {
            let value = groups[[i, j]].to_f64().unwrap();
            let rounded_value = (value * 1000.0).round() as i64;
            unique_values[j].insert(rounded_value);
        }
    }

    // Generate subgroups for single attributes
    for (group_idx, group_name) in group_names.iter().enumerate() {
        for &int_value in &unique_values[group_idx] {
            let value = int_value as f64 / 1000.0;
            let subgroup_name = format!("{}={}", group_name, value);

            // Extract data for this subgroup
            let mut subgroup_y_true = Vec::new();
            let mut subgroup_y_pred = Vec::new();

            for i in 0..n_samples {
                let group_value = groups[[i, group_idx]].to_f64().unwrap();
                let rounded_value = (group_value * 1000.0).round() as i64;

                if rounded_value == int_value {
                    subgroup_y_true.push(y_true[i].clone());
                    subgroup_y_pred.push(y_pred[i].clone());
                }
            }

            // Compute metric for this subgroup
            if !subgroup_y_true.is_empty() {
                let subgroup_metric = metric_fn(&subgroup_y_true, &subgroup_y_pred);
                results.insert(subgroup_name, subgroup_metric);
            }
        }
    }

    // Generate intersectional subgroups if there's more than one group
    if n_groups > 1 {
        // Generate all possible combinations of group values
        generate_intersectional_subgroups(
            y_true,
            y_pred,
            groups,
            group_names,
            &unique_values,
            &mut results,
            metric_fn,
        )?;
    }

    Ok(results)
}

// Helper function to generate intersectional subgroups
fn generate_intersectional_subgroups<T, S1, S2, S3, F>(
    y_true: &ArrayBase<S1, Ix1>,
    y_pred: &ArrayBase<S2, Ix1>,
    groups: &ArrayBase<S3, Ix2>,
    group_names: &[String],
    unique_values: &[BTreeSet<i64>],
    results: &mut HashMap<String, f64>,
    metric_fn: F,
) -> Result<()>
where
    T: Real + PartialOrd + Clone,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    S3: Data<Elem = T>,
    F: Fn(&[T], &[T]) -> f64,
{
    let n_samples = y_true.len();
    let n_groups = groups.ncols();

    // Generate all pairwise intersections
    for i in 0..n_groups {
        for j in (i + 1)..n_groups {
            for &int_value_i in &unique_values[i] {
                for &int_value_j in &unique_values[j] {
                    let value_i = int_value_i as f64 / 1000.0;
                    let value_j = int_value_j as f64 / 1000.0;

                    let subgroup_name = format!(
                        "{}={} & {}={}",
                        group_names[i], value_i, group_names[j], value_j
                    );

                    // Extract data for this intersectional subgroup
                    let mut subgroup_y_true = Vec::new();
                    let mut subgroup_y_pred = Vec::new();

                    for k in 0..n_samples {
                        let group_value_i = groups[[k, i]].to_f64().unwrap();
                        let rounded_value_i = (group_value_i * 1000.0).round() as i64;

                        let group_value_j = groups[[k, j]].to_f64().unwrap();
                        let rounded_value_j = (group_value_j * 1000.0).round() as i64;

                        if rounded_value_i == int_value_i && rounded_value_j == int_value_j {
                            subgroup_y_true.push(y_true[k].clone());
                            subgroup_y_pred.push(y_pred[k].clone());
                        }
                    }

                    // Compute metric for this intersectional subgroup
                    if !subgroup_y_true.is_empty() {
                        let subgroup_metric = metric_fn(&subgroup_y_true, &subgroup_y_pred);
                        results.insert(subgroup_name, subgroup_metric);
                    }
                }
            }
        }
    }

    // If needed, generate higher-order intersections for 3+ groups
    if n_groups > 2 {
        // This implementation only does pairwise intersections
        // Higher-order intersections could be added using a recursive approach
    }

    Ok(())
}

/// Computes fairness metrics for intersection of protected groups.
///
/// This function calculates fairness metrics (such as demographic parity, equalized odds)
/// for intersectional subgroups, revealing how bias may affect people with multiple
/// protected attributes differently than those with a single protected attribute.
///
/// # Arguments
///
/// * `y_true` - Ground truth labels
/// * `y_pred` - Predicted labels or scores
/// * `protected_features` - Matrix where each column is a protected attribute
/// * `feature_names` - Names of the protected attributes
///
/// # Returns
///
/// * HashMap mapping intersectional groups to their fairness metrics
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array2};
/// use scirs2_metrics::fairness::bias_detection::intersectional_fairness;
///
/// // Create sample dataset
/// let y_true = array![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
/// let y_pred = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0];
///
/// // Protected attributes: gender (0=male, 1=female) and race (0=group A, 1=group B)
/// let protected_features = Array2::from_shape_vec((8, 2), vec![
///     0.0, 0.0,
///     0.0, 1.0,
///     1.0, 0.0,
///     1.0, 0.0,
///     0.0, 1.0,
///     0.0, 0.0,
///     1.0, 1.0,
///     1.0, 1.0,
/// ]).unwrap();
///
/// let feature_names = vec!["gender".to_string(), "race".to_string()];
///
/// // Analyze intersectional fairness
/// let results = intersectional_fairness(
///     &y_true,
///     &y_pred,
///     &protected_features,
///     &feature_names
/// ).unwrap();
///
/// // Examine fairness metrics across intersectional groups
/// for (group, metrics) in results {
///     println!("Group: {}", group);
///     println!("  Demographic Parity: {:.3}", metrics.demographic_parity);
///     println!("  Equalized Odds: {:.3}", metrics.equalized_odds);
///     println!("  Equal Opportunity: {:.3}", metrics.equal_opportunity);
/// }
/// ```
#[derive(Debug, Clone)]
/// Metrics for measuring different aspects of fairness
pub struct FairnessMetrics {
    /// Demographic parity score - measures equal prediction rate across groups
    pub demographic_parity: f64,
    /// Equalized odds score - measures equal error rates across groups
    pub equalized_odds: f64,
    /// Equal opportunity score - measures equal true positive rates across groups
    pub equal_opportunity: f64,
}

/// Evaluates fairness metrics across intersections of multiple protected attributes.
///
/// This function computes fairness metrics (demographic parity, equalized odds, equal opportunity)
/// at the intersections of different protected attributes, which helps identify cases where
/// bias may affect specific subgroups that share multiple protected characteristics.
///
/// # Arguments
///
/// * `y_true` - Ground truth labels
/// * `y_pred` - Predicted labels or scores
/// * `protected_features` - A matrix where each column represents a different protected attribute
/// * `feature_names` - Names of the protected attributes (corresponding to columns in protected_features)
///
/// # Returns
///
/// * A HashMap mapping intersection names (e.g., "gender=0,race=1") to fairness metric sets
pub fn intersectional_fairness<T, S1, S2, S3>(
    y_true: &ArrayBase<S1, Ix1>,
    y_pred: &ArrayBase<S2, Ix1>,
    protected_features: &ArrayBase<S3, Ix2>,
    feature_names: &[String],
) -> Result<HashMap<String, FairnessMetrics>>
where
    T: Real + PartialOrd + Clone,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    S3: Data<Elem = T>,
{
    // Check dimensions
    let n_samples = y_true.len();
    if n_samples != y_pred.len() || n_samples != protected_features.nrows() {
        return Err(MetricsError::InvalidInput(format!(
            "Dimensions mismatch: y_true ({}), y_pred ({}), protected_features ({} rows)",
            n_samples,
            y_pred.len(),
            protected_features.nrows()
        )));
    }

    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty arrays provided".to_string(),
        ));
    }

    let n_protected = protected_features.ncols();
    if n_protected != feature_names.len() {
        return Err(MetricsError::InvalidInput(format!(
            "Number of protected feature columns ({}) doesn't match number of feature names ({})",
            n_protected,
            feature_names.len()
        )));
    }

    // Verify data contains at least one sample
    if n_samples == 0 {
        return Err(MetricsError::InvalidInput("Empty input data".to_string()));
    }

    let mut results = HashMap::new();

    // Find unique values for each protected feature
    let mut unique_values = vec![BTreeSet::new(); n_protected];
    for i in 0..n_samples {
        for j in 0..n_protected {
            let value = protected_features[[i, j]].to_f64().unwrap();
            let rounded_value = (value * 1000.0).round() as i64;
            unique_values[j].insert(rounded_value);
        }
    }

    // Analyze single protected attributes
    for (feat_idx, feat_name) in feature_names.iter().enumerate() {
        for &int_value in &unique_values[feat_idx] {
            let value = int_value as f64 / 1000.0;

            // Create binary protected group mask for this attribute value
            let mut protected_group = vec![T::zero(); n_samples];

            for i in 0..n_samples {
                let feat_value = protected_features[[i, feat_idx]].to_f64().unwrap();
                let rounded_value = (feat_value * 1000.0).round() as i64;

                if rounded_value == int_value {
                    protected_group[i] = T::one();
                }
            }

            // Skip if all or none of the samples are in this group
            let num_in_group: usize = protected_group.iter().filter(|&&x| x > T::zero()).count();
            if num_in_group == 0 || num_in_group == n_samples {
                continue;
            }

            // Create ndarray from protected_group Vec
            let protected_group_array = ndarray::Array::from(protected_group);

            // Calculate fairness metrics
            let group_name = format!("{}={}", feat_name, value);
            let metrics = calculate_fairness_metrics(y_true, y_pred, &protected_group_array)?;
            results.insert(group_name, metrics);
        }
    }

    // Analyze intersectional protected attributes
    if n_protected > 1 {
        for i in 0..n_protected {
            for j in (i + 1)..n_protected {
                for &int_value_i in &unique_values[i] {
                    for &int_value_j in &unique_values[j] {
                        let value_i = int_value_i as f64 / 1000.0;
                        let value_j = int_value_j as f64 / 1000.0;

                        // Create binary protected group mask for this intersection
                        let mut protected_group = vec![T::zero(); n_samples];

                        for k in 0..n_samples {
                            let feat_i_value = protected_features[[k, i]].to_f64().unwrap();
                            let rounded_i = (feat_i_value * 1000.0).round() as i64;

                            let feat_j_value = protected_features[[k, j]].to_f64().unwrap();
                            let rounded_j = (feat_j_value * 1000.0).round() as i64;

                            if rounded_i == int_value_i && rounded_j == int_value_j {
                                protected_group[k] = T::one();
                            }
                        }

                        // Skip if all or none of the samples are in this intersectional group
                        let num_in_group: usize =
                            protected_group.iter().filter(|&&x| x > T::zero()).count();
                        if num_in_group == 0 || num_in_group == n_samples {
                            continue;
                        }

                        // Calculate fairness metrics for this intersection
                        let group_name = format!(
                            "{}={} & {}={}",
                            feature_names[i], value_i, feature_names[j], value_j
                        );

                        // Create ndarray from protected_group Vec
                        let protected_group_array = ndarray::Array::from(protected_group);

                        let metrics =
                            calculate_fairness_metrics(y_true, y_pred, &protected_group_array)?;
                        results.insert(group_name, metrics);
                    }
                }
            }
        }
    }

    Ok(results)
}

// Helper function to calculate all fairness metrics
fn calculate_fairness_metrics<T, S1, S2, S3>(
    y_true: &ArrayBase<S1, Ix1>,
    y_pred: &ArrayBase<S2, Ix1>,
    protected_group: &ArrayBase<S3, Ix1>,
) -> Result<FairnessMetrics>
where
    T: Real + PartialOrd + Clone,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    S3: Data<Elem = T>,
{
    use crate::fairness::{
        demographic_parity_difference, equal_opportunity_difference, equalized_odds_difference,
    };

    // Calculate demographic parity
    let dp = demographic_parity_difference(y_pred, protected_group)?;

    // Calculate equalized odds and equal opportunity if we have ground truth labels
    let eod = equalized_odds_difference(y_true, y_pred, protected_group).unwrap_or(1.0);

    let eo = equal_opportunity_difference(y_true, y_pred, protected_group).unwrap_or(1.0);

    Ok(FairnessMetrics {
        demographic_parity: dp,
        equalized_odds: eod,
        equal_opportunity: eo,
    })
}
