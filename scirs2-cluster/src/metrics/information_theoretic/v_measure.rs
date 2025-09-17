//! V-Measure and related clustering evaluation metrics
//!
//! This module provides the V-measure score and its components (homogeneity and
//! completeness) for evaluating clustering quality. These metrics are based on
//! information theory and measure different aspects of clustering performance.

use ndarray::ArrayView1;
use num_traits::{Float, FromPrimitive};
use std::collections::HashMap;
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};

/// Calculate V-measure score between two label assignments
///
/// V-measure is the harmonic mean of homogeneity and completeness scores.
/// It provides a single score that balances both aspects of clustering quality.
///
/// # Arguments
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Predicted cluster labels
///
/// # Returns
/// * `Result<F>` - The V-measure score (0 to 1, higher is better)
///
/// # Example
/// ```rust
/// use ndarray::Array1;
/// use scirs2_cluster::metrics::v_measure_score;
///
/// let true_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
/// let pred_labels = Array1::from_vec(vec![0, 0, 1, 1, 1, 2]);
///
/// let v_measure: f64 = v_measure_score(true_labels.view(), pred_labels.view()).unwrap();
/// assert!(v_measure >= 0.0 && v_measure <= 1.0);
/// ```
pub fn v_measure_score<F>(labels_true: ArrayView1<i32>, labels_pred: ArrayView1<i32>) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let homogeneity = homogeneity_score(labels_true, labels_pred)?;
    let completeness = completeness_score(labels_true, labels_pred)?;

    if homogeneity + completeness == F::zero() {
        Ok(F::zero())
    } else {
        let two = F::from(2).unwrap();
        Ok(two * homogeneity * completeness / (homogeneity + completeness))
    }
}

/// Calculate homogeneity score
///
/// Homogeneity measures whether each cluster contains only members of a single class.
/// A clustering is homogeneous if all clusters contain only data points from one class.
/// It is calculated as H(C|K) where C is the true classes and K is the predicted clusters.
///
/// # Arguments
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Predicted cluster labels
///
/// # Returns
/// * `Result<F>` - The homogeneity score (0 to 1, higher is better)
///
/// # Example
/// ```rust
/// use ndarray::Array1;
/// use scirs2_cluster::metrics::homogeneity_score;
///
/// let true_labels = Array1::from_vec(vec![0, 0, 1, 1]);
/// let pred_labels = Array1::from_vec(vec![0, 0, 1, 1]);
///
/// let homogeneity: f64 = homogeneity_score(true_labels.view(), pred_labels.view()).unwrap();
/// assert!((homogeneity - 1.0).abs() < 1e-10); // Perfect homogeneity
/// ```
pub fn homogeneity_score<F>(labels_true: ArrayView1<i32>, labels_pred: ArrayView1<i32>) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    if labels_true.len() != labels_pred.len() {
        return Err(ClusteringError::InvalidInput(
            "True and predicted labels must have the same length".to_string(),
        ));
    }

    let h_true = entropy::<F>(labels_true)?;
    if h_true == F::zero() {
        return Ok(F::one());
    }

    let h_true_given_pred = conditional_entropy::<F>(labels_true, labels_pred)?;
    Ok((h_true - h_true_given_pred) / h_true)
}

/// Calculate completeness score
///
/// Completeness measures whether all members of a given class are assigned
/// to the same cluster. A clustering is complete if all data points from
/// one class are in the same cluster.
///
/// # Arguments
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Predicted cluster labels
///
/// # Returns
/// * `Result<F>` - The completeness score (0 to 1, higher is better)
///
/// # Example
/// ```rust
/// use ndarray::Array1;
/// use scirs2_cluster::metrics::completeness_score;
///
/// let true_labels = Array1::from_vec(vec![0, 0, 1, 1]);
/// let pred_labels = Array1::from_vec(vec![0, 0, 1, 1]);
///
/// let completeness: f64 = completeness_score(true_labels.view(), pred_labels.view()).unwrap();
/// assert!((completeness - 1.0).abs() < 1e-10); // Perfect completeness
/// ```
pub fn completeness_score<F>(
    labels_true: ArrayView1<i32>,
    labels_pred: ArrayView1<i32>,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    if labels_true.len() != labels_pred.len() {
        return Err(ClusteringError::InvalidInput(
            "True and predicted labels must have the same length".to_string(),
        ));
    }

    let h_pred = entropy::<F>(labels_pred)?;
    if h_pred == F::zero() {
        return Ok(F::one());
    }

    let h_pred_given_true = conditional_entropy::<F>(labels_pred, labels_true)?;
    Ok((h_pred - h_pred_given_true) / h_pred)
}

/// Calculate homogeneity, completeness, and V-measure in one call
///
/// This is more efficient than calling each function separately as it avoids
/// redundant calculations.
///
/// # Arguments
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Predicted cluster labels
///
/// # Returns
/// * `Result<(F, F, F)>` - Tuple of (homogeneity, completeness, v_measure)
pub fn homogeneity_completeness_v_measure<F>(
    labels_true: ArrayView1<i32>,
    labels_pred: ArrayView1<i32>,
) -> Result<(F, F, F)>
where
    F: Float + FromPrimitive + Debug,
{
    if labels_true.len() != labels_pred.len() {
        return Err(ClusteringError::InvalidInput(
            "True and predicted labels must have the same length".to_string(),
        ));
    }

    let h_true = entropy::<F>(labels_true)?;
    let h_pred = entropy::<F>(labels_pred)?;

    // Handle degenerate cases
    if h_true == F::zero() && h_pred == F::zero() {
        return Ok((F::one(), F::one(), F::one()));
    }
    if h_true == F::zero() {
        return Ok((F::one(), F::zero(), F::zero()));
    }
    if h_pred == F::zero() {
        return Ok((F::zero(), F::one(), F::zero()));
    }

    let h_true_given_pred = conditional_entropy::<F>(labels_true, labels_pred)?;
    let h_pred_given_true = conditional_entropy::<F>(labels_pred, labels_true)?;

    let homogeneity = (h_true - h_true_given_pred) / h_true;
    let completeness = (h_pred - h_pred_given_true) / h_pred;

    let v_measure = if homogeneity + completeness == F::zero() {
        F::zero()
    } else {
        let two = F::from(2).unwrap();
        two * homogeneity * completeness / (homogeneity + completeness)
    };

    Ok((homogeneity, completeness, v_measure))
}

/// Calculate weighted V-measure with custom beta parameter
///
/// This variant allows controlling the relative weight given to homogeneity
/// versus completeness in the harmonic mean calculation.
///
/// # Arguments
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Predicted cluster labels
/// * `beta` - Weight parameter (beta=1 gives equal weight, beta>1 favors completeness)
///
/// # Returns
/// * `Result<F>` - The weighted V-measure score
pub fn weighted_v_measure_score<F>(
    labels_true: ArrayView1<i32>,
    labels_pred: ArrayView1<i32>,
    beta: F,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let homogeneity = homogeneity_score(labels_true, labels_pred)?;
    let completeness = completeness_score(labels_true, labels_pred)?;

    if homogeneity == F::zero() && completeness == F::zero() {
        return Ok(F::zero());
    }

    let beta_squared = beta * beta;
    let numerator = (F::one() + beta_squared) * homogeneity * completeness;
    let denominator = beta_squared * homogeneity + completeness;

    if denominator == F::zero() {
        Ok(F::zero())
    } else {
        Ok(numerator / denominator)
    }
}

/// Calculate entropy of a label assignment
fn entropy<F>(labels: ArrayView1<i32>) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let mut counts = HashMap::new();
    for &label in labels.iter() {
        *counts.entry(label).or_insert(0) += 1;
    }

    let n_samples = labels.len();
    if n_samples == 0 {
        return Ok(F::zero());
    }

    let n_samples_f = F::from(n_samples).unwrap();
    let mut entropy = F::zero();

    for &count in counts.values() {
        if count > 0 {
            let p = F::from(count).unwrap() / n_samples_f;
            entropy = entropy - p * p.ln();
        }
    }

    Ok(entropy)
}

/// Calculate conditional entropy H(X|Y)
fn conditional_entropy<F>(x_labels: ArrayView1<i32>, y_labels: ArrayView1<i32>) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let n_samples = x_labels.len();
    if n_samples == 0 {
        return Ok(F::zero());
    }

    // Build joint and marginal counts
    let mut joint_counts = HashMap::new();
    let mut y_counts = HashMap::new();

    for i in 0..n_samples {
        let x = x_labels[i];
        let y = y_labels[i];

        *joint_counts.entry((x, y)).or_insert(0) += 1;
        *y_counts.entry(y).or_insert(0) += 1;
    }

    let n_samples_f = F::from(n_samples).unwrap();
    let mut conditional_entropy = F::zero();

    for (&(x, y), &n_xy) in &joint_counts {
        let n_y = y_counts[&y];

        let p_xy = F::from(n_xy).unwrap() / n_samples_f;
        let p_x_given_y = F::from(n_xy).unwrap() / F::from(n_y).unwrap();

        if p_xy > F::zero() && p_x_given_y > F::zero() {
            conditional_entropy = conditional_entropy - p_xy * p_x_given_y.ln();
        }
    }

    Ok(conditional_entropy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_v_measure_perfect() {
        let labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        let v_measure: f64 = v_measure_score(labels.view(), labels.view()).unwrap();
        assert!((v_measure - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_homogeneity_perfect() {
        let true_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        let pred_labels = Array1::from_vec(vec![0, 1, 2, 3, 4, 5]); // Each point in its own cluster

        let homogeneity: f64 = homogeneity_score(true_labels.view(), pred_labels.view()).unwrap();
        assert!((homogeneity - 1.0).abs() < 1e-10); // Perfect homogeneity
    }

    #[test]
    fn test_completeness_perfect() {
        let true_labels = Array1::from_vec(vec![0, 1, 2, 3, 4, 5]);
        let pred_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]); // All points in same clusters

        let completeness: f64 = completeness_score(true_labels.view(), pred_labels.view()).unwrap();
        assert!((completeness - 1.0).abs() < 1e-10); // Perfect completeness
    }

    #[test]
    fn test_homogeneity_completeness_v_measure() {
        let true_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        let pred_labels = Array1::from_vec(vec![0, 0, 1, 1, 1, 2]);

        let (h, c, v): (f64, f64, f64) =
            homogeneity_completeness_v_measure(true_labels.view(), pred_labels.view()).unwrap();

        assert!(h >= 0.0 && h <= 1.0);
        assert!(c >= 0.0 && c <= 1.0);
        assert!(v >= 0.0 && v <= 1.0);

        // V-measure should be harmonic mean of homogeneity and completeness
        let expected_v = if h + c > 0.0 {
            2.0 * h * c / (h + c)
        } else {
            0.0
        };
        assert!((v - expected_v).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_v_measure() {
        let true_labels = Array1::from_vec(vec![0, 0, 1, 1]);
        let pred_labels = Array1::from_vec(vec![0, 1, 0, 1]);

        // Test different beta values
        let v1: f64 =
            weighted_v_measure_score(true_labels.view(), pred_labels.view(), 1.0).unwrap();
        let v2: f64 =
            weighted_v_measure_score(true_labels.view(), pred_labels.view(), 2.0).unwrap();

        assert!(v1 >= 0.0 && v1 <= 1.0);
        assert!(v2 >= 0.0 && v2 <= 1.0);

        // With beta=1, should equal regular V-measure
        let regular_v: f64 = v_measure_score(true_labels.view(), pred_labels.view()).unwrap();
        assert!((v1 - regular_v).abs() < 1e-10);
    }

    #[test]
    fn test_entropy() {
        // Uniform distribution should have maximum entropy
        let uniform = Array1::from_vec(vec![0, 1, 2, 3]);
        let h: f64 = entropy::<f64>(uniform.view()).unwrap();
        let expected = 4.0_f64.ln();
        assert!((h - expected).abs() < 1e-10);

        // Single value should have zero entropy
        let single = Array1::from_vec(vec![0, 0, 0, 0]);
        let h_single: f64 = entropy::<f64>(single.view()).unwrap();
        assert!(h_single.abs() < 1e-10);
    }

    #[test]
    fn test_conditional_entropy() {
        // Perfect dependence should give zero conditional entropy
        let x = Array1::from_vec(vec![0, 0, 1, 1]);
        let y = Array1::from_vec(vec![0, 0, 1, 1]); // Same as x

        let h_x_given_y: f64 = conditional_entropy::<f64>(x.view(), y.view()).unwrap();
        assert!(h_x_given_y.abs() < 1e-10);
    }

    #[test]
    fn test_degenerate_cases() {
        // Empty arrays
        let empty = Array1::from_vec(vec![]);
        let (h, c, v): (f64, f64, f64) =
            homogeneity_completeness_v_measure(empty.view(), empty.view()).unwrap();
        assert_eq!(h, 1.0);
        assert_eq!(c, 1.0);
        assert_eq!(v, 1.0);

        // Single element
        let single_true = Array1::from_vec(vec![0]);
        let single_pred = Array1::from_vec(vec![1]);
        let (h, c, v): (f64, f64, f64) =
            homogeneity_completeness_v_measure(single_true.view(), single_pred.view()).unwrap();
        assert_eq!(h, 1.0);
        assert_eq!(c, 1.0);
        assert_eq!(v, 1.0);
    }
}
