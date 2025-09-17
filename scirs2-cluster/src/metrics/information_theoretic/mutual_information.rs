//! Mutual Information metrics for clustering evaluation
//!
//! This module provides various mutual information-based metrics for evaluating
//! clustering quality when ground truth labels are available. These metrics
//! measure how well the predicted clustering preserves the information structure
//! of the true class labels.

use ndarray::{Array1, ArrayView1};
use num_traits::{Float, FromPrimitive};
use std::collections::HashMap;
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};

/// Calculate mutual information between two label assignments
///
/// Mutual Information (MI) measures the amount of information shared between
/// two random variables. In clustering, it measures how much information
/// the predicted clusters share with the true clusters.
///
/// # Arguments
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Predicted cluster labels
///
/// # Returns
/// * `Result<F>` - The mutual information score (higher is better)
///
/// # Example
/// ```rust
/// use ndarray::Array1;
/// use scirs2_cluster::metrics::mutual_info_score;
///
/// let true_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
/// let pred_labels = Array1::from_vec(vec![0, 0, 1, 1, 1, 2]);
///
/// let mi: f64 = mutual_info_score(true_labels.view(), pred_labels.view()).unwrap();
/// assert!(mi > 0.0);
/// ```
pub fn mutual_info_score<F>(labels_true: ArrayView1<i32>, labels_pred: ArrayView1<i32>) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    if labels_true.len() != labels_pred.len() {
        return Err(ClusteringError::InvalidInput(
            "True and predicted labels must have the same length".to_string(),
        ));
    }

    let n_samples = labels_true.len();
    if n_samples == 0 {
        return Ok(F::zero());
    }

    // Create contingency table
    let contingency = build_contingency_table(labels_true, labels_pred);
    let mut mi = F::zero();
    let n_samples_f = F::from(n_samples).unwrap();

    // Calculate marginal probabilities
    let mut row_sums = HashMap::new();
    let mut col_sums = HashMap::new();
    for (&(i, j), &count) in &contingency {
        *row_sums.entry(i).or_insert(0) += count;
        *col_sums.entry(j).or_insert(0) += count;
    }

    // Calculate mutual information
    for (&(i, j), &n_ij) in &contingency {
        if n_ij > 0 {
            let n_i = row_sums[&i];
            let n_j = col_sums[&j];
            let p_ij = F::from(n_ij).unwrap() / n_samples_f;
            let p_i = F::from(n_i).unwrap() / n_samples_f;
            let p_j = F::from(n_j).unwrap() / n_samples_f;

            mi = mi + p_ij * (p_ij / (p_i * p_j)).ln();
        }
    }

    Ok(mi)
}

/// Calculate normalized mutual information between two label assignments
///
/// Normalized Mutual Information (NMI) normalizes the mutual information
/// by the geometric mean of the entropies of both label assignments.
/// This makes the score range between 0 and 1, where 1 indicates perfect
/// agreement and 0 indicates no mutual information.
///
/// # Arguments
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Predicted cluster labels
///
/// # Returns
/// * `Result<F>` - The normalized mutual information score (0 to 1, higher is better)
pub fn normalized_mutual_info_score<F>(
    labels_true: ArrayView1<i32>,
    labels_pred: ArrayView1<i32>,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let mi = mutual_info_score::<F>(labels_true, labels_pred)?;

    if mi == F::zero() {
        return Ok(F::zero());
    }

    let h_true = entropy::<F>(labels_true)?;
    let h_pred = entropy::<F>(labels_pred)?;

    if h_true == F::zero() || h_pred == F::zero() {
        return Ok(F::zero());
    }

    let normalizer = (h_true * h_pred).sqrt();
    Ok(mi / normalizer)
}

/// Calculate adjusted mutual information between two label assignments
///
/// Adjusted Mutual Information (AMI) adjusts the mutual information score
/// for chance, similar to how the adjusted rand index works. It accounts
/// for the expected mutual information between random clusterings.
///
/// # Arguments
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Predicted cluster labels
///
/// # Returns
/// * `Result<F>` - The adjusted mutual information score (higher is better)
pub fn adjusted_mutual_info_score<F>(
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

    let n_samples = labels_true.len();
    if n_samples <= 1 {
        return Ok(F::zero());
    }

    let mi = mutual_info_score::<F>(labels_true, labels_pred)?;
    let h_true = entropy::<F>(labels_true)?;
    let h_pred = entropy::<F>(labels_pred)?;

    // Calculate expected mutual information
    let expected_mi = calculate_expected_mutual_information::<F>(labels_true, labels_pred)?;

    // Adjusted MI formula
    let denominator = (h_true + h_pred) / F::from(2).unwrap() - expected_mi;

    if denominator.abs() < F::from(1e-10).unwrap() {
        Ok(F::zero())
    } else {
        Ok((mi - expected_mi) / denominator)
    }
}

/// Calculate normalized mutual information with different normalization methods
///
/// This function provides flexibility in choosing the normalization method
/// for mutual information calculation.
///
/// # Arguments
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Predicted cluster labels
/// * `method` - Normalization method ("arithmetic", "geometric", "min", "max")
///
/// # Returns
/// * `Result<F>` - The normalized mutual information score
pub fn normalized_mutual_info_score_with_method<F>(
    labels_true: ArrayView1<i32>,
    labels_pred: ArrayView1<i32>,
    method: &str,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let mi = mutual_info_score::<F>(labels_true, labels_pred)?;

    if mi == F::zero() {
        return Ok(F::zero());
    }

    let h_true = entropy::<F>(labels_true)?;
    let h_pred = entropy::<F>(labels_pred)?;

    if h_true == F::zero() || h_pred == F::zero() {
        return Ok(F::zero());
    }

    let normalizer = match method {
        "arithmetic" => (h_true + h_pred) / F::from(2).unwrap(),
        "geometric" => (h_true * h_pred).sqrt(),
        "min" => h_true.min(h_pred),
        "max" => h_true.max(h_pred),
        _ => {
            return Err(ClusteringError::InvalidInput(format!(
                "Unknown normalization method: {}",
                method
            )))
        }
    };

    if normalizer == F::zero() {
        Ok(F::zero())
    } else {
        Ok(mi / normalizer)
    }
}

/// Calculate conditional mutual information I(X; Y | Z)
///
/// Conditional mutual information measures the mutual information between
/// X and Y given knowledge of Z. This is useful for analyzing clustering
/// performance in the presence of confounding variables.
///
/// # Arguments
/// * `x_labels` - First set of labels
/// * `y_labels` - Second set of labels
/// * `z_labels` - Conditioning labels
///
/// # Returns
/// * `Result<F>` - The conditional mutual information score
pub fn conditional_mutual_information<F>(
    x_labels: ArrayView1<i32>,
    y_labels: ArrayView1<i32>,
    z_labels: ArrayView1<i32>,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    if x_labels.len() != y_labels.len() || x_labels.len() != z_labels.len() {
        return Err(ClusteringError::InvalidInput(
            "All label arrays must have the same length".to_string(),
        ));
    }

    let n_samples = x_labels.len();
    if n_samples == 0 {
        return Ok(F::zero());
    }

    let n_samples_f = F::from(n_samples).unwrap();

    // Build joint contingency tables
    let mut xyz_counts = HashMap::new();
    let mut xz_counts = HashMap::new();
    let mut yz_counts = HashMap::new();
    let mut z_counts = HashMap::new();

    for i in 0..n_samples {
        let x = x_labels[i];
        let y = y_labels[i];
        let z = z_labels[i];

        *xyz_counts.entry((x, y, z)).or_insert(0) += 1;
        *xz_counts.entry((x, z)).or_insert(0) += 1;
        *yz_counts.entry((y, z)).or_insert(0) += 1;
        *z_counts.entry(z).or_insert(0) += 1;
    }

    // Calculate conditional mutual information
    let mut cmi = F::zero();

    for (&(x, y, z), &n_xyz) in &xyz_counts {
        if n_xyz > 0 {
            let n_xz = xz_counts[&(x, z)];
            let n_yz = yz_counts[&(y, z)];
            let n_z = z_counts[&z];

            let p_xyz = F::from(n_xyz).unwrap() / n_samples_f;
            let p_xz = F::from(n_xz).unwrap() / n_samples_f;
            let p_yz = F::from(n_yz).unwrap() / n_samples_f;
            let p_z = F::from(n_z).unwrap() / n_samples_f;

            if p_xz > F::zero() && p_yz > F::zero() && p_z > F::zero() {
                cmi = cmi + p_xyz * ((p_xyz * p_z) / (p_xz * p_yz)).ln();
            }
        }
    }

    Ok(cmi)
}

/// Build contingency table from two label arrays
fn build_contingency_table(
    labels_true: ArrayView1<i32>,
    labels_pred: ArrayView1<i32>,
) -> HashMap<(i32, i32), usize> {
    let mut contingency = HashMap::new();
    for (&true_label, &pred_label) in labels_true.iter().zip(labels_pred.iter()) {
        *contingency.entry((true_label, pred_label)).or_insert(0) += 1;
    }
    contingency
}

/// Calculate entropy of a label assignment
pub fn entropy<F>(labels: ArrayView1<i32>) -> Result<F>
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

/// Calculate expected mutual information between random clusterings
fn calculate_expected_mutual_information<F>(
    labels_true: ArrayView1<i32>,
    labels_pred: ArrayView1<i32>,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let n_samples = labels_true.len();
    if n_samples <= 1 {
        return Ok(F::zero());
    }

    let contingency = build_contingency_table(labels_true, labels_pred);

    // Calculate marginal sums
    let mut a = Vec::new(); // true cluster sizes
    let mut b = Vec::new(); // pred cluster sizes

    let mut true_counts = HashMap::new();
    let mut pred_counts = HashMap::new();

    for &label in labels_true.iter() {
        *true_counts.entry(label).or_insert(0usize) += 1;
    }
    for &label in labels_pred.iter() {
        *pred_counts.entry(label).or_insert(0usize) += 1;
    }

    for &count in true_counts.values() {
        a.push(count);
    }
    for &count in pred_counts.values() {
        b.push(count);
    }

    let n_samples_f = F::from(n_samples).unwrap();
    let mut emi = F::zero();

    // Calculate expected mutual information using the formula
    for &a_i in &a {
        for &b_j in &b {
            let start: usize = (a_i + b_j).saturating_sub(n_samples as usize);
            let end = a_i.min(b_j);

            for n_ij in start..=end {
                if n_ij > 0 {
                    let p_ij = F::from(n_ij).unwrap() / n_samples_f;
                    let p_i = F::from(a_i).unwrap() / n_samples_f;
                    let p_j = F::from(b_j).unwrap() / n_samples_f;

                    // Hypergeometric probability approximation
                    let prob = hypergeometric_pmf(n_ij, n_samples, a_i, b_j);

                    if prob > 0.0 && p_ij > F::zero() {
                        emi = emi + F::from(prob).unwrap() * p_ij * (p_ij / (p_i * p_j)).ln();
                    }
                }
            }
        }
    }

    Ok(emi)
}

/// Hypergeometric probability mass function approximation
fn hypergeometric_pmf(k: usize, n: usize, k_success: usize, n_draws: usize) -> f64 {
    if k > k_success || k > n_draws || (n_draws - k) > (n - k_success) {
        return 0.0;
    }

    // Simple approximation - in practice would use a more robust implementation
    let numerator =
        binomial_coefficient(k_success, k) * binomial_coefficient(n - k_success, n_draws - k);
    let denominator = binomial_coefficient(n, n_draws);

    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Simple binomial coefficient approximation
fn binomial_coefficient(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    if k == 0 || k == n {
        return 1.0;
    }

    let k = k.min(n - k); // Take advantage of symmetry
    let mut result = 1.0;

    for i in 0..k {
        result *= (n - i) as f64 / (i + 1) as f64;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_mutual_info_score_perfect_match() {
        let labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        let mi: f64 = mutual_info_score::<f64>(labels.view(), labels.view()).unwrap();

        // MI should equal entropy for perfect match
        let h: f64 = entropy::<f64>(labels.view()).unwrap();
        assert!((mi - h).abs() < 1e-10);
    }

    #[test]
    fn test_normalized_mutual_info_score() {
        let true_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        let pred_labels = Array1::from_vec(vec![0, 0, 1, 1, 1, 2]);

        let nmi: f64 =
            normalized_mutual_info_score::<f64>(true_labels.view(), pred_labels.view()).unwrap();
        assert!(nmi >= 0.0 && nmi <= 1.0);
    }

    #[test]
    fn test_adjusted_mutual_info_score() {
        let true_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        let pred_labels = Array1::from_vec(vec![0, 0, 1, 1, 1, 2]);

        let ami: f64 =
            adjusted_mutual_info_score::<f64>(true_labels.view(), pred_labels.view()).unwrap();
        assert!(ami >= -1.0 && ami <= 1.0);
    }

    #[test]
    fn test_entropy() {
        let uniform_labels = Array1::from_vec(vec![0, 1, 2, 3]);
        let h: f64 = entropy::<f64>(uniform_labels.view()).unwrap();

        // Entropy of uniform distribution should be log(n)
        let expected = 4.0_f64.ln();
        assert!((h - expected).abs() < 1e-10);
    }

    #[test]
    fn test_build_contingency_table() {
        let true_labels = Array1::from_vec(vec![0, 0, 1, 1]);
        let pred_labels = Array1::from_vec(vec![0, 1, 0, 1]);

        let contingency = build_contingency_table(true_labels.view(), pred_labels.view());
        assert_eq!(contingency.len(), 4);
        assert_eq!(contingency[&(0, 0)], 1);
        assert_eq!(contingency[&(0, 1)], 1);
        assert_eq!(contingency[&(1, 0)], 1);
        assert_eq!(contingency[&(1, 1)], 1);
    }

    #[test]
    fn test_conditional_mutual_information() {
        let x = Array1::from_vec(vec![0, 0, 1, 1]);
        let y = Array1::from_vec(vec![0, 1, 0, 1]);
        let z = Array1::from_vec(vec![0, 0, 1, 1]);

        let cmi: f64 = conditional_mutual_information::<f64>(x.view(), y.view(), z.view()).unwrap();
        assert!(cmi.is_finite());
    }
}
