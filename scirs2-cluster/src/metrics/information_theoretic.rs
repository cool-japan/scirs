//! Information-theoretic clustering evaluation metrics
//!
//! This module provides information-theoretic metrics for evaluating clustering algorithms
//! when ground truth labels are available. These metrics are based on concepts from
//! information theory and measure how well the clustering preserves the information
//! in the true class structure.

use ndarray::{Array1, ArrayView1};
use num_traits::{Float, FromPrimitive};
use rand::prelude::*;
use std::collections::HashMap;
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};
use ndarray::Array2;
use rand::seq::SliceRandom;
use scirs2_core::parallel_ops::*;
use scirs2_core::Rng;

/// Calculate mutual information between two label assignments
///
/// Mutual Information (MI) measures the amount of information shared between
/// two random variables. In clustering, it measures how much information
/// the predicted clusters share with the true clusters.
///
/// # Arguments
///
/// * `labels_true` - Ground truth class labels
/// * `labelspred` - Predicted cluster labels
///
/// # Returns
///
/// The mutual information score (higher is better)
///
/// # Example
///
/// ```
/// use ndarray::Array1;
/// use scirs2_cluster::metrics::mutual_info_score;
///
/// let true_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
/// let pred_labels = Array1::from_vec(vec![0, 0, 1, 1, 1, 2]);
///
/// let mi = mutual_info_score(true_labels.view(), pred_labels.view()).unwrap();
/// assert!(mi > 0.0);
/// ```
#[allow(dead_code)]
pub fn mutual_info_score<F>(labels_true: ArrayView1<i32>, labelspred: ArrayView1<i32>) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    if labels_true.len() != labelspred.len() {
        return Err(ClusteringError::InvalidInput(
            "True and predicted labels must have the same length".to_string(),
        ));
    }

    let n_samples = labels_true.len();
    if n_samples == 0 {
        return Ok(F::zero());
    }

    // Create contingency table
    let contingency = build_contingency_table(labels_true, labelspred);

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
/// This provides a score between 0 and 1, where 1 indicates perfect agreement.
///
/// # Arguments
///
/// * `labels_true` - Ground truth class labels
/// * `labelspred` - Predicted cluster labels
///
/// # Returns
///
/// The normalized mutual information score (0 to 1, higher is better)
#[allow(dead_code)]
pub fn normalized_mutual_info_score<F>(
    labels_true: ArrayView1<i32>,
    labelspred: ArrayView1<i32>,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let mi = mutual_info_score::<F>(labels_true, labelspred)?;
    let h_true = entropy::<F>(labels_true)?;
    let h_pred = entropy::<F>(labelspred)?;

    if h_true == F::zero() && h_pred == F::zero() {
        return Ok(F::one());
    }

    let normalizer = (h_true * h_pred).sqrt();

    if normalizer == F::zero() {
        Ok(F::zero())
    } else {
        Ok(mi / normalizer)
    }
}

/// Calculate adjusted mutual information between two label assignments
///
/// Adjusted Mutual Information (AMI) adjusts the mutual information for chance,
/// providing a score that is corrected for random labeling.
///
/// # Arguments
///
/// * `labels_true` - Ground truth class labels
/// * `labelspred` - Predicted cluster labels
///
/// # Returns
///
/// The adjusted mutual information score (higher is better, can be negative)
#[allow(dead_code)]
pub fn adjusted_mutual_info_score<F>(
    labels_true: ArrayView1<i32>,
    labelspred: ArrayView1<i32>,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let mi = mutual_info_score::<F>(labels_true, labelspred)?;
    let emi = expected_mutual_info::<F>(labels_true, labelspred)?;
    let h_true = entropy::<F>(labels_true)?;
    let h_pred = entropy::<F>(labelspred)?;

    let max_entropy = if h_true > h_pred { h_true } else { h_pred };

    if max_entropy == emi {
        Ok(F::zero())
    } else {
        Ok((mi - emi) / (max_entropy - emi))
    }
}

/// Calculate adjusted rand index between two label assignments
///
/// The Adjusted Rand Index (ARI) measures the similarity between two clusterings
/// by considering all pairs of samples and counting pairs that are assigned
/// in the same or different clusters in both clusterings.
///
/// # Arguments
///
/// * `labels_true` - Ground truth class labels
/// * `labelspred` - Predicted cluster labels
///
/// # Returns
///
/// The adjusted rand index (-1 to 1, higher is better)
#[allow(dead_code)]
pub fn adjusted_rand_score<F>(
    labels_true: ArrayView1<i32>,
    labelspred: ArrayView1<i32>,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    if labels_true.len() != labelspred.len() {
        return Err(ClusteringError::InvalidInput(
            "True and predicted labels must have the same length".to_string(),
        ));
    }

    let n_samples = labels_true.len();
    if n_samples == 0 {
        return Ok(F::one());
    }

    let contingency = build_contingency_table(labels_true, labelspred);

    // Calculate marginal sums
    let mut row_sums = HashMap::new();
    let mut col_sums = HashMap::new();

    for (&(i, j), &count) in &contingency {
        *row_sums.entry(i).or_insert(0) += count;
        *col_sums.entry(j).or_insert(0) += count;
    }

    // Calculate ARI components
    let mut sum_comb_c = F::zero();
    for &count in contingency.values() {
        if count >= 2 {
            sum_comb_c = sum_comb_c + F::from(comb2(count)).unwrap();
        }
    }

    let mut sum_comb_a = F::zero();
    for &count in row_sums.values() {
        if count >= 2 {
            sum_comb_a = sum_comb_a + F::from(comb2(count)).unwrap();
        }
    }

    let mut sum_comb_b = F::zero();
    for &count in col_sums.values() {
        if count >= 2 {
            sum_comb_b = sum_comb_b + F::from(comb2(count)).unwrap();
        }
    }

    let n_choose_2 = F::from(comb2(n_samples)).unwrap();
    let expected_index = sum_comb_a * sum_comb_b / n_choose_2;
    let max_index = (sum_comb_a + sum_comb_b) / F::from(2).unwrap();

    if max_index == expected_index {
        Ok(F::zero())
    } else {
        Ok((sum_comb_c - expected_index) / (max_index - expected_index))
    }
}

/// Calculate V-measure score
///
/// V-measure is the harmonic mean of homogeneity and completeness.
/// It provides a single score that balances both aspects of clustering quality.
///
/// # Arguments
///
/// * `labels_true` - Ground truth class labels
/// * `labelspred` - Predicted cluster labels
///
/// # Returns
///
/// The V-measure score (0 to 1, higher is better)
#[allow(dead_code)]
pub fn v_measure_score<F>(labels_true: ArrayView1<i32>, labelspred: ArrayView1<i32>) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let homogeneity = homogeneity_score(labels_true, labelspred)?;
    let completeness = completeness_score(labels_true, labelspred)?;

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
///
/// # Arguments
///
/// * `labels_true` - Ground truth class labels
/// * `labelspred` - Predicted cluster labels
///
/// # Returns
///
/// The homogeneity score (0 to 1, higher is better)
#[allow(dead_code)]
pub fn homogeneity_score<F>(labels_true: ArrayView1<i32>, labelspred: ArrayView1<i32>) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let h_true = entropy::<F>(labels_true)?;

    if h_true == F::zero() {
        return Ok(F::one());
    }

    let h_true_given_pred = conditional_entropy::<F>(labels_true, labelspred)?;
    Ok((h_true - h_true_given_pred) / h_true)
}

/// Calculate completeness score
///
/// Completeness measures whether all members of a given class are assigned
/// to the same cluster. A clustering is complete if all data points from
/// one class are in the same cluster.
///
/// # Arguments
///
/// * `labels_true` - Ground truth class labels
/// * `labelspred` - Predicted cluster labels
///
/// # Returns
///
/// The completeness score (0 to 1, higher is better)
#[allow(dead_code)]
pub fn completeness_score<F>(labels_true: ArrayView1<i32>, labelspred: ArrayView1<i32>) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let h_pred = entropy::<F>(labelspred)?;

    if h_pred == F::zero() {
        return Ok(F::one());
    }

    let h_pred_given_true = conditional_entropy::<F>(labelspred, labels_true)?;
    Ok((h_pred - h_pred_given_true) / h_pred)
}

// Helper functions

/// Build contingency table from two label arrays
#[allow(dead_code)]
fn build_contingency_table(
    labels_true: ArrayView1<i32>,
    labelspred: ArrayView1<i32>,
) -> HashMap<(i32, i32), usize> {
    let mut contingency = HashMap::new();

    for (&true_label, &pred_label) in labels_true.iter().zip(labelspred.iter()) {
        *contingency.entry((true_label, pred_label)).or_insert(0) += 1;
    }

    contingency
}

/// Calculate entropy of a label assignment
#[allow(dead_code)]
fn entropy<F>(labels: ArrayView1<i32>) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let mut counts = HashMap::new();
    for &label in labels.iter() {
        *counts.entry(label).or_insert(0) += 1;
    }

    let n_samples = labels.len();
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
#[allow(dead_code)]
fn conditional_entropy<F>(labelsx: ArrayView1<i32>, labelsy: ArrayView1<i32>) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let contingency = build_contingency_table(labelsx, labelsy);

    // Calculate marginal counts for Y
    let mut y_counts = HashMap::new();
    for (&(_, y), &count) in &contingency {
        *y_counts.entry(y).or_insert(0) += count;
    }

    let n_samples = labelsx.len();
    let n_samples_f = F::from(n_samples).unwrap();

    let mut cond_entropy = F::zero();

    for (&y, &ny) in &y_counts {
        if ny == 0 {
            continue;
        }

        let py = F::from(ny).unwrap() / n_samples_f;
        let mut hx_giveny = F::zero();

        for (&(x, y_val), &nxy) in &contingency {
            if y_val == y && nxy > 0 {
                let px_giveny = F::from(nxy).unwrap() / F::from(ny).unwrap();
                hx_giveny = hx_giveny - px_giveny * px_giveny.ln();
            }
        }

        cond_entropy = cond_entropy + py * hx_giveny;
    }

    Ok(cond_entropy)
}

/// Calculate expected mutual information for random labeling
#[allow(dead_code)]
fn expected_mutual_info<F>(labels_true: ArrayView1<i32>, labelspred: ArrayView1<i32>) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    // Proper calculation of Expected Mutual Information using hypergeometric distribution
    let contingency = build_contingency_table(labels_true, labelspred);

    let n = labels_true.len() as f64;
    let mut row_sums = HashMap::new();
    let mut col_sums = HashMap::new();

    // Calculate marginals
    for ((i, j), &count) in &contingency {
        *row_sums.entry(i).or_insert(0) += count;
        *col_sums.entry(j).or_insert(0) += count;
    }

    let mut emi = 0.0;

    // For each cell in the contingency table, compute expected mutual information
    for &a_i in row_sums.values() {
        for &b_j in col_sums.values() {
            let a_i_f = a_i as f64;
            let b_j_f = b_j as f64;

            // Calculate range of possible values for n_ij under hypergeometric distribution
            let start = (a_i_f + b_j_f - n).max(0.0) as usize;
            let end = a_i.min(b_j) + 1;

            for n_ij in start..end {
                if n_ij == 0 {
                    continue;
                }

                let n_ij_f = n_ij as f64;

                // Calculate hypergeometric probability P(n_ij | a_i, b_j, n)
                let prob = hypergeometric_pmf(n_ij, n as usize, a_i, b_j);

                if prob > 0.0 {
                    // Calculate mutual information contribution for this cell
                    let mi_contrib = n_ij_f * (n * n_ij_f / (a_i_f * b_j_f)).ln();
                    emi += prob * mi_contrib;
                }
            }
        }
    }

    Ok(F::from(emi / (n * (2.0_f64).ln())).unwrap())
}

/// Calculate combinations C(n, 2) = n * (n-1) / 2
#[allow(dead_code)]
fn comb2(n: usize) -> usize {
    if n < 2 {
        0
    } else {
        n * (n - 1) / 2
    }
}

/// Calculate hypergeometric probability mass function
/// P(X = k) = C(K, k) * C(N-K, n-k) / C(N, n)
/// where N is population size, K is success states in population,
/// n is sample size, k is observed successes
#[allow(dead_code)]
fn hypergeometric_pmf(k: usize, n_total: usize, row_sum: usize, colsum: usize) -> f64 {
    if k > row_sum || k > colsum || (row_sum + colsum - k) > n_total {
        return 0.0;
    }

    // Use log space to avoid overflow
    let log_prob =
        log_comb(row_sum, k) + log_comb(n_total - row_sum, colsum - k) - log_comb(n_total, colsum);
    log_prob.exp()
}

/// Calculate log of binomial coefficient C(n, k) = log(n!) - log(k!) - log((n-k)!)
#[allow(dead_code)]
fn log_comb(n: usize, k: usize) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    if k == 0 || k == n {
        return 0.0;
    }

    let k = k.min(n - k); // Use symmetry property

    let mut result = 0.0;
    for i in 0..k {
        result += ((n - i) as f64).ln() - ((i + 1) as f64).ln();
    }

    result
}

/// Enumeration of normalization methods for Normalized Mutual Information
#[derive(Debug, Clone, Copy)]
pub enum NormalizationMethod {
    /// Geometric mean normalization: sqrt(H(X) * H(Y))
    Geometric,
    /// Arithmetic mean normalization: (H(X) + H(Y)) / 2
    Arithmetic,
    /// Minimum normalization: min(H(X), H(Y))
    Min,
    /// Maximum normalization: max(H(X), H(Y))
    Max,
}

/// Calculate normalized mutual information with different normalization methods
///
/// This provides more flexibility than the standard NMI implementation by supporting
/// multiple normalization strategies as used in sklearn.
///
/// # Arguments
///
/// * `labels_true` - Ground truth class labels
/// * `labelspred` - Predicted cluster labels
/// * `method` - Normalization method to use
///
/// # Returns
///
/// The normalized mutual information score (0 to 1, higher is better)
#[allow(dead_code)]
pub fn normalized_mutual_info_score_with_method<F>(
    labels_true: ArrayView1<i32>,
    labelspred: ArrayView1<i32>,
    method: NormalizationMethod,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let mi: F = mutual_info_score(labels_true, labelspred)?;
    let h_true: F = entropy(labels_true)?;
    let h_pred: F = entropy(labelspred)?;

    let normalizer: F = match method {
        NormalizationMethod::Geometric => {
            let product: F = h_true * h_pred;
            product.sqrt()
        }
        NormalizationMethod::Arithmetic => (h_true + h_pred) / F::from(2).unwrap(),
        NormalizationMethod::Min => h_true.min(h_pred),
        NormalizationMethod::Max => h_true.max(h_pred),
    };

    if normalizer == F::zero() {
        Ok(F::zero())
    } else {
        Ok(mi / normalizer)
    }
}

/// Calculate Fowlkes-Mallows Index
///
/// The Fowlkes-Mallows Index is the geometric mean of precision and recall
/// for cluster assignments. It's equivalent to the F1 score for clustering.
///
/// # Arguments
///
/// * `labels_true` - Ground truth class labels
/// * `labelspred` - Predicted cluster labels
///
/// # Returns
///
/// The Fowlkes-Mallows Index (0 to 1, higher is better)
#[allow(dead_code)]
pub fn fowlkes_mallows_score<F>(
    labels_true: ArrayView1<i32>,
    labelspred: ArrayView1<i32>,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    if labels_true.len() != labelspred.len() {
        return Err(ClusteringError::InvalidInput(
            "True and predicted labels must have the same length".to_string(),
        ));
    }

    let n = labels_true.len();
    let mut tp = 0; // True positives: same cluster in both
    let mut fp = 0; // False positives: same cluster in pred, different in _true
    let mut fn_count = 0; // False negatives: same cluster in true, different in _pred

    for i in 0..n {
        for j in (i + 1)..n {
            let same_true = labels_true[i] == labels_true[j];
            let same_pred = labelspred[i] == labelspred[j];

            match (same_true, same_pred) {
                (true, true) => tp += 1,
                (false, true) => fp += 1,
                (true, false) => fn_count += 1,
                (false, false) => {} // True negative, not used in FM calculation
            }
        }
    }

    let precision = if tp + fp == 0 {
        F::zero()
    } else {
        F::from(tp as f64 / (tp + fp) as f64).unwrap()
    };

    let recall = if tp + fn_count == 0 {
        F::zero()
    } else {
        F::from(tp as f64 / (tp + fn_count) as f64).unwrap()
    };

    // Fowlkes-Mallows Index is the geometric mean of precision and recall
    if precision == F::zero() || recall == F::zero() {
        Ok(F::zero())
    } else {
        Ok((precision * recall).sqrt())
    }
}

/// Calculate Rand Index (unadjusted)
///
/// The Rand Index measures the similarity between two clusterings by considering
/// all pairs of samples and counting pairs that are assigned to the same or different
/// clusters in both clusterings.
///
/// # Arguments
///
/// * `labels_true` - Ground truth class labels
/// * `labelspred` - Predicted cluster labels
///
/// # Returns
///
/// The Rand Index (0 to 1, higher is better)
#[allow(dead_code)]
pub fn rand_score<F>(labels_true: ArrayView1<i32>, labelspred: ArrayView1<i32>) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    if labels_true.len() != labelspred.len() {
        return Err(ClusteringError::InvalidInput(
            "True and predicted labels must have the same length".to_string(),
        ));
    }

    let n = labels_true.len();
    let mut agreements = 0; // Pairs classified the same way in both clusterings

    for i in 0..n {
        for j in (i + 1)..n {
            let same_true = labels_true[i] == labels_true[j];
            let same_pred = labelspred[i] == labelspred[j];

            if same_true == same_pred {
                agreements += 1;
            }
        }
    }

    let total_pairs = n * (n - 1) / 2;
    if total_pairs == 0 {
        Ok(F::one())
    } else {
        Ok(F::from(agreements as f64 / total_pairs as f64).unwrap())
    }
}

/// Calculate Normalized Variation of Information
///
/// This normalizes the Variation of Information to the range [0, 1],
/// making it easier to interpret and compare across different datasets.
///
/// # Arguments
///
/// * `labels_true` - Ground truth class labels
/// * `labelspred` - Predicted cluster labels
///
/// # Returns
///
/// The normalized variation of information (0 to 1, lower is better)
#[allow(dead_code)]
pub fn normalized_variation_of_information<F>(
    labels_true: ArrayView1<i32>,
    labelspred: ArrayView1<i32>,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let h_true = entropy::<F>(labels_true)?;
    let h_pred = entropy::<F>(labelspred)?;
    let h_cond_true_pred = conditional_entropy::<F>(labels_true, labelspred)?;
    let h_cond_pred_true = conditional_entropy::<F>(labelspred, labels_true)?;

    let vi = h_cond_true_pred + h_cond_pred_true;
    let joint_entropy = h_true + h_cond_pred_true;

    if joint_entropy == F::zero() {
        Ok(F::zero())
    } else {
        Ok(vi / joint_entropy)
    }
}

/// Calculate Jensen-Shannon Divergence between two label distributions
///
/// The Jensen-Shannon Divergence is a symmetric measure of divergence between
/// two probability distributions, based on the Kullback-Leibler divergence.
///
/// # Arguments
///
/// * `labels_true` - Ground truth class labels
/// * `labelspred` - Predicted cluster labels
///
/// # Returns
///
/// The Jensen-Shannon Divergence (0 to 1, lower is better)
#[allow(dead_code)]
pub fn jensen_shannon_divergence<F>(
    labels_true: ArrayView1<i32>,
    labelspred: ArrayView1<i32>,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    // Convert labels to probability distributions
    let dist_true = labels_to_distribution::<F>(labels_true)?;
    let dist_pred = labels_to_distribution::<F>(labelspred)?;

    // Ensure both distributions have the same support
    let mut all_labels = std::collections::HashSet::new();
    for &label in labels_true.iter() {
        all_labels.insert(label);
    }
    for &label in labelspred.iter() {
        all_labels.insert(label);
    }

    let mut sorted_labels: Vec<_> = all_labels.into_iter().collect();
    sorted_labels.sort();

    let mut p = Vec::new();
    let mut q = Vec::new();

    for &label in &sorted_labels {
        p.push(*dist_true.get(&label).unwrap_or(&F::zero()));
        q.push(*dist_pred.get(&label).unwrap_or(&F::zero()));
    }

    // Calculate Jensen-Shannon Divergence
    let mut js_div = F::zero();
    let ln2 = F::from(2.0_f64.ln()).unwrap();

    for i in 0..p.len() {
        let pi = p[i];
        let qi = q[i];
        let mi = (pi + qi) / F::from(2).unwrap();

        if pi > F::zero() && mi > F::zero() {
            js_div = js_div + pi * (pi / mi).ln() / F::from(2).unwrap();
        }
        if qi > F::zero() && mi > F::zero() {
            js_div = js_div + qi * (qi / mi).ln() / F::from(2).unwrap();
        }
    }

    // Normalize by log(2) to get range [0, 1]
    Ok(js_div / ln2)
}

/// Convert label array to probability distribution
#[allow(dead_code)]
fn labels_to_distribution<F>(labels: ArrayView1<i32>) -> Result<HashMap<i32, F>>
where
    F: Float + FromPrimitive + Debug,
{
    let mut counts = HashMap::new();
    for &label in labels.iter() {
        *counts.entry(label).or_insert(0) += 1;
    }

    let total = labels.len() as f64;
    let mut distribution = HashMap::new();
    for (label, count) in counts {
        distribution.insert(label, F::from(count as f64 / total).unwrap());
    }

    Ok(distribution)
}

/// Calculate Conditional Mutual Information I(X; Y | Z)
///
/// This measures the mutual information between X and Y given knowledge of Z.
/// Useful for analyzing multi-level clustering relationships.
///
/// # Arguments
///
/// * `x_labels` - First variable labels
/// * `y_labels` - Second variable labels  
/// * `z_labels` - Conditioning variable labels
///
/// # Returns
///
/// The conditional mutual information
#[allow(dead_code)]
pub fn conditional_mutual_information<F>(
    x_labels: ArrayView1<i32>,
    y_labels: ArrayView1<i32>,
    z_labels: ArrayView1<i32>,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    if x_labels.len() != y_labels.len() || y_labels.len() != z_labels.len() {
        return Err(ClusteringError::InvalidInput(
            "All label arrays must have the same length".to_string(),
        ));
    }

    // Build three-way contingency table
    let mut triplet_counts = HashMap::new();
    let mut z_counts = HashMap::new();
    let mut xz_counts = HashMap::new();
    let mut yz_counts = HashMap::new();

    for i in 0..x_labels.len() {
        let x = x_labels[i];
        let y = y_labels[i];
        let z = z_labels[i];

        *triplet_counts.entry((x, y, z)).or_insert(0) += 1;
        *z_counts.entry(z).or_insert(0) += 1;
        *xz_counts.entry((x, z)).or_insert(0) += 1;
        *yz_counts.entry((y, z)).or_insert(0) += 1;
    }

    let n = x_labels.len() as f64;
    let mut cmi = F::zero();

    // Calculate I(X; Y | Z) = sum over x,y,z of P(x,y,z) * log(P(x,y,z) * P(z) / (P(x,z) * P(y,z)))
    for ((x, y, z), &nxyz) in &triplet_counts {
        let pxyz = nxyz as f64 / n;
        let p_z = *z_counts.get(z).unwrap() as f64 / n;
        let pxz = *xz_counts.get(&(*x, *z)).unwrap() as f64 / n;
        let pyz = *yz_counts.get(&(*y, *z)).unwrap() as f64 / n;

        if pxyz > 0.0 && p_z > 0.0 && pxz > 0.0 && pyz > 0.0 {
            let log_term = (pxyz * p_z / (pxz * pyz)).ln();
            cmi = cmi + F::from(pxyz * log_term).unwrap();
        }
    }

    Ok(cmi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_mutual_information() {
        // Perfect clustering
        let true_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        let pred_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);

        let mi: f64 = mutual_info_score(true_labels.view(), pred_labels.view()).unwrap();
        assert!(mi > 0.0);

        // Random clustering should have lower MI
        let random_labels = Array1::from_vec(vec![0, 1, 0, 1, 0, 1]);
        let mi_random: f64 = mutual_info_score(true_labels.view(), random_labels.view()).unwrap();
        assert!(mi > mi_random);
    }

    #[test]
    fn test_normalized_mutual_info() {
        let true_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        let pred_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);

        let nmi: f64 =
            normalized_mutual_info_score(true_labels.view(), pred_labels.view()).unwrap();
        assert!(nmi >= 0.0 && nmi <= 1.0);
        assert!(nmi > 0.9); // Should be close to 1 for perfect clustering
    }

    #[test]
    fn test_adjusted_rand_score() {
        let true_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        let pred_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);

        let ari: f64 = adjusted_rand_score(true_labels.view(), pred_labels.view()).unwrap();
        assert!(ari > 0.9); // Should be close to 1 for perfect clustering

        // Test with random clustering
        let random_labels = Array1::from_vec(vec![0, 1, 0, 1, 0, 1]);
        let ari_random: f64 =
            adjusted_rand_score(true_labels.view(), random_labels.view()).unwrap();
        assert!(ari > ari_random);
    }

    #[test]
    fn test_v_measure() {
        let true_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        let pred_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);

        let v_measure: f64 = v_measure_score(true_labels.view(), pred_labels.view()).unwrap();
        assert!(v_measure >= 0.0 && v_measure <= 1.0);
        assert!(v_measure > 0.9); // Should be close to 1 for perfect clustering
    }

    #[test]
    fn test_homogeneity_and_completeness() {
        let true_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        let pred_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);

        let homogeneity: f64 = homogeneity_score(true_labels.view(), pred_labels.view()).unwrap();
        let completeness: f64 = completeness_score(true_labels.view(), pred_labels.view()).unwrap();

        assert!(homogeneity >= 0.0 && homogeneity <= 1.0);
        assert!(completeness >= 0.0 && completeness <= 1.0);
        assert!(homogeneity > 0.9);
        assert!(completeness > 0.9);
    }

    #[test]
    fn test_empty_labels() {
        let empty_labels = Array1::from_vec(vec![]);
        let empty_labels2 = Array1::from_vec(vec![]);

        let mi: f64 = mutual_info_score(empty_labels.view(), empty_labels2.view()).unwrap();
        assert_eq!(mi, 0.0);
    }

    #[test]
    fn test_single_cluster() {
        let true_labels = Array1::from_vec(vec![0, 0, 0, 0]);
        let pred_labels = Array1::from_vec(vec![1, 1, 1, 1]);

        let nmi: f64 =
            normalized_mutual_info_score(true_labels.view(), pred_labels.view()).unwrap();
        assert_eq!(nmi, 1.0); // Single clusters should have perfect NMI
    }

    #[test]
    fn test_conditional_mutual_information() {
        let x_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        let y_labels = Array1::from_vec(vec![0, 1, 0, 1, 0, 1]);
        let z_labels = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);

        let cmi: f64 =
            conditional_mutual_information(x_labels.view(), y_labels.view(), z_labels.view())
                .unwrap();
        assert!(cmi >= 0.0); // CMI should be non-negative
    }

    #[test]
    fn test_jensen_shannon_divergence() {
        let true_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        let pred_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);

        let jsd: f64 = jensen_shannon_divergence(true_labels.view(), pred_labels.view()).unwrap();
        assert!(jsd >= 0.0 && jsd <= 1.0);
        assert!(jsd < 0.1); // Should be close to 0 for identical clusterings
    }

    #[test]
    fn test_fowlkes_mallows_score() {
        let true_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        let pred_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);

        let fm: f64 = fowlkes_mallows_score(true_labels.view(), pred_labels.view()).unwrap();
        assert!(fm >= 0.0 && fm <= 1.0);
        assert!(fm > 0.9); // Should be close to 1 for perfect clustering
    }
}

/// Advanced information-theoretic clustering validation methods
pub mod advanced_validation {
    use super::*;
    use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
    use scirs2_core::parallel_ops::*;
    use std::collections::HashMap;

    /// Stability-based clustering validation using bootstrap resampling
    pub struct StabilityValidator<F: Float + FromPrimitive + Debug + Send + Sync> {
        /// Number of bootstrap samples
        n_bootstrap: usize,
        /// Fraction of data to sample in each bootstrap
        sample_fraction: f64,
        /// Random seed for reproducibility
        seed: u64,
        /// Phantom marker for float type
        _phantom: std::marker::PhantomData<F>,
    }

    impl<F: Float + FromPrimitive + Debug + Send + Sync> StabilityValidator<F> {
        /// Create a new stability validator
        pub fn new(n_bootstrap: usize, sample_fraction: f64) -> Self {
            use rand::Rng;
            let mut thread_rng = rand::rngs::StdRng::seed_from_u64(42); // Use fixed seed for reproducibility
            Self {
                n_bootstrap,
                sample_fraction,
                seed: thread_rng.random::<u64>(),
                _phantom: std::marker::PhantomData,
            }
        }

        /// Compute clustering stability using bootstrap consensus
        pub fn compute_stability<ClusterFn>(
            &mut self,
            data: ArrayView2<F>,
            cluster_fn: ClusterFn,
        ) -> Result<StabilityMetrics>
        where
            ClusterFn: Fn(ArrayView2<F>) -> Result<Array1<usize>> + Sync + Send,
            F: 'static,
        {
            let n_samples = data.nrows();
            let sample_size = (n_samples as f64 * self.sample_fraction) as usize;

            // Generate bootstrap samples and compute clusterings
            let clusterings: Vec<Array1<usize>> = (0..self.n_bootstrap)
                .into_par_iter()
                .map(|_| {
                    // Create bootstrap sample indices
                    let mut rng = rand::rng();
                    let indices: Vec<usize> = (0..sample_size)
                        .map(|_| rng.random_range(0..n_samples))
                        .collect();

                    // Extract bootstrap sample
                    let mut bootstrapdata = Array2::zeros((sample_size, data.ncols()));
                    for (i, &idx) in indices.iter().enumerate() {
                        bootstrapdata.row_mut(i).assign(&data.row(idx));
                    }

                    // Apply clustering algorithm
                    cluster_fn(bootstrapdata.view())
                })
                .collect::<Result<Vec<_>>>()?;

            // Compute pairwise agreement between clusterings
            let agreements = self.compute_pairwise_agreements(&clusterings)?;

            // Calculate stability metrics
            let mean_agreement = agreements.iter().sum::<f64>() / agreements.len() as f64;
            let agreement_variance = agreements
                .iter()
                .map(|&x| (x - mean_agreement).powi(2))
                .sum::<f64>()
                / agreements.len() as f64;

            let stability_index = mean_agreement;
            let confidence_interval = self.compute_confidence_interval(&agreements, 0.95)?;

            Ok(StabilityMetrics {
                stability_index,
                mean_agreement,
                agreement_variance,
                confidence_interval,
                n_bootstrap: self.n_bootstrap,
                clusterings_analyzed: clusterings.len(),
            })
        }

        /// Compute pairwise agreement between clusterings using adjusted rand index
        fn compute_pairwise_agreements(&self, clusterings: &[Array1<usize>]) -> Result<Vec<f64>> {
            let n_clusterings = clusterings.len();
            let mut agreements = Vec::new();

            for i in 0..n_clusterings {
                for j in (i + 1)..n_clusterings {
                    // Convert to i32 for adjusted_rand_score
                    let labels1: Array1<i32> = clusterings[i].mapv(|x| x as i32);
                    let labels2: Array1<i32> = clusterings[j].mapv(|x| x as i32);

                    let agreement = adjusted_rand_score::<f64>(labels1.view(), labels2.view())?;
                    agreements.push(agreement);
                }
            }

            Ok(agreements)
        }

        /// Compute confidence interval for stability metric
        fn compute_confidence_interval(
            &self,
            agreements: &[f64],
            confidence_level: f64,
        ) -> Result<(f64, f64)> {
            if agreements.is_empty() {
                return Ok((0.0, 0.0));
            }

            let mut sorted_agreements = agreements.to_vec();
            sorted_agreements.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let alpha = 1.0 - confidence_level;
            let lower_idx = ((alpha / 2.0) * sorted_agreements.len() as f64) as usize;
            let upper_idx = ((1.0 - alpha / 2.0) * sorted_agreements.len() as f64) as usize;

            let lower_bound = sorted_agreements
                .get(lower_idx)
                .copied()
                .unwrap_or(sorted_agreements[0]);
            let upper_bound = sorted_agreements
                .get(upper_idx.min(sorted_agreements.len() - 1))
                .copied()
                .unwrap_or(sorted_agreements[sorted_agreements.len() - 1]);

            Ok((lower_bound, upper_bound))
        }

        /// Assess clustering stability across different parameter settings
        pub fn parameter_stability_analysis<ClusterFn>(
            &mut self,
            data: ArrayView2<F>,
            parameter_sets: &[HashMap<String, f64>],
            cluster_fn: ClusterFn,
        ) -> Result<Vec<ParameterStabilityResult>>
        where
            ClusterFn:
                Fn(ArrayView2<F>, &HashMap<String, f64>) -> Result<Array1<usize>> + Sync + Send,
            F: 'static,
        {
            parameter_sets
                .par_iter()
                .map(|params| {
                    let clusterings: Vec<Array1<usize>> = (0..self.n_bootstrap)
                        .into_par_iter()
                        .map(|_| {
                            let n_samples = data.nrows();
                            let sample_size = (n_samples as f64 * self.sample_fraction) as usize;
                            let mut rng =
                                rand::rngs::StdRng::seed_from_u64(self.seed.wrapping_add(1));

                            let indices: Vec<usize> = (0..sample_size)
                                .map(|_| rng.random_range(0..n_samples))
                                .collect();

                            let mut bootstrapdata = Array2::zeros((sample_size, data.ncols()));
                            for (i, &idx) in indices.iter().enumerate() {
                                bootstrapdata.row_mut(i).assign(&data.row(idx));
                            }

                            cluster_fn(bootstrapdata.view(), params)
                        })
                        .collect::<Result<Vec<_>>>()?;

                    let agreements = self.compute_pairwise_agreements(&clusterings)?;
                    let stability = agreements.iter().sum::<f64>() / agreements.len() as f64;

                    Ok(ParameterStabilityResult {
                        parameters: params.clone(),
                        stability_score: stability,
                        n_valid_clusterings: clusterings.len(),
                    })
                })
                .collect()
        }
    }

    /// Results from stability analysis
    #[derive(Debug, Clone)]
    pub struct StabilityMetrics {
        /// Overall stability index (0 to 1, higher is better)
        pub stability_index: f64,
        /// Mean agreement between clusterings
        pub mean_agreement: f64,
        /// Variance in agreement scores
        pub agreement_variance: f64,
        /// Confidence interval for stability
        pub confidence_interval: (f64, f64),
        /// Number of bootstrap samples used
        pub n_bootstrap: usize,
        /// Number of clusterings analyzed
        pub clusterings_analyzed: usize,
    }

    /// Results from parameter stability analysis
    #[derive(Debug, Clone)]
    pub struct ParameterStabilityResult {
        /// Parameter set used
        pub parameters: HashMap<String, f64>,
        /// Stability score for this parameter set
        pub stability_score: f64,
        /// Number of valid clusterings produced
        pub n_valid_clusterings: usize,
    }

    /// Advanced information-theoretic metrics for clustering validation
    pub struct AdvancedInformationMetrics;

    impl AdvancedInformationMetrics {
        /// Calculate Transfer Entropy between two clustering solutions
        pub fn transfer_entropy<F>(
            source_labels: ArrayView1<i32>,
            target_labels: ArrayView1<i32>,
            lag: usize,
        ) -> Result<F>
        where
            F: Float + FromPrimitive + Debug,
        {
            if source_labels.len() != target_labels.len() {
                return Err(ClusteringError::InvalidInput(
                    "Source and target _labels must have the same length".to_string(),
                ));
            }

            if lag >= source_labels.len() {
                return Err(ClusteringError::InvalidInput(
                    "Lag cannot be greater than or equal to sequence length".to_string(),
                ));
            }

            let n_samples = source_labels.len() - lag;
            let mut triplet_counts = HashMap::new();
            let mut target_future_counts = HashMap::new();
            let mut target_past_counts = HashMap::new();

            // Build transition counts with lag
            for i in lag..source_labels.len() {
                let source_past = source_labels[i - lag];
                let target_past = target_labels[i - lag];
                let target_future = target_labels[i];

                *triplet_counts
                    .entry((source_past, target_past, target_future))
                    .or_insert(0) += 1;
                *target_future_counts.entry(target_future).or_insert(0) += 1;
                *target_past_counts.entry(target_past).or_insert(0) += 1;
            }

            let n_samples_f = F::from(n_samples).unwrap();
            let mut transfer_entropy = F::zero();

            // Calculate H(Y_t+1 | Y_t) - H(Y_t+1 | Y_t, X_t)
            for (&(_source_past, target_past, target_future), &count) in &triplet_counts {
                let pxyz = F::from(count).unwrap() / n_samples_f;
                let p_z = F::from(*target_future_counts.get(&target_future).unwrap()).unwrap()
                    / n_samples_f;
                let py =
                    F::from(*target_past_counts.get(&target_past).unwrap()).unwrap() / n_samples_f;

                // Find joint probability P(Y_t, Y_t+1)
                let mut yz_count = 0;
                for i in lag..source_labels.len() {
                    if target_labels[i - lag] == target_past && target_labels[i] == target_future {
                        yz_count += 1;
                    }
                }
                let pyz = F::from(yz_count).unwrap() / n_samples_f;

                if pxyz > F::zero() && p_z > F::zero() && py > F::zero() && pyz > F::zero() {
                    transfer_entropy = transfer_entropy + pxyz * (pxyz * py / (pyz * p_z)).ln();
                }
            }

            Ok(transfer_entropy)
        }

        /// Calculate Multi-Information (Total Correlation) for multiple clustering solutions
        pub fn multi_information<F>(_clusteringsolutions: &[ArrayView1<i32>]) -> Result<F>
        where
            F: Float + FromPrimitive + Debug,
        {
            if _clusteringsolutions.is_empty() {
                return Ok(F::zero());
            }

            let n_samples = _clusteringsolutions[0].len();
            for solution in _clusteringsolutions {
                if solution.len() != n_samples {
                    return Err(ClusteringError::InvalidInput(
                        "All clustering _solutions must have the same length".to_string(),
                    ));
                }
            }

            // Calculate individual entropies
            let mut individual_entropies = Vec::new();
            for solution in _clusteringsolutions {
                let entropy = entropy::<F>(*solution)?;
                individual_entropies.push(entropy);
            }

            // Calculate joint entropy
            let joint_entropy = Self::joint_entropy(_clusteringsolutions)?;

            // Multi-information = sum of individual entropies - joint entropy
            let sum_individual: F = individual_entropies
                .iter()
                .copied()
                .fold(F::zero(), |acc, x| acc + x);
            Ok(sum_individual - joint_entropy)
        }

        /// Calculate joint entropy for multiple clustering solutions
        fn joint_entropy<F>(_clusteringsolutions: &[ArrayView1<i32>]) -> Result<F>
        where
            F: Float + FromPrimitive + Debug,
        {
            let n_samples = _clusteringsolutions[0].len();
            let mut joint_counts = HashMap::new();

            // Count joint occurrences
            for i in 0..n_samples {
                let joint_label: Vec<i32> = _clusteringsolutions
                    .iter()
                    .map(|solution| solution[i])
                    .collect();
                *joint_counts.entry(joint_label).or_insert(0) += 1;
            }

            let n_samples_f = F::from(n_samples).unwrap();
            let mut joint_entropy = F::zero();

            for &count in joint_counts.values() {
                if count > 0 {
                    let p = F::from(count).unwrap() / n_samples_f;
                    joint_entropy = joint_entropy - p * p.ln();
                }
            }

            Ok(joint_entropy)
        }

        /// Calculate Interaction Information for three clustering solutions
        pub fn interaction_information<F>(
            labels_a: ArrayView1<i32>,
            labels_b: ArrayView1<i32>,
            labels_c: ArrayView1<i32>,
        ) -> Result<F>
        where
            F: Float + FromPrimitive + Debug,
        {
            if labels_a.len() != labels_b.len() || labels_b.len() != labels_c.len() {
                return Err(ClusteringError::InvalidInput(
                    "All label arrays must have the same length".to_string(),
                ));
            }

            // I(A; B; C) = I(A; B|C) - I(A; B)
            let conditional_mi = conditional_mutual_information::<F>(labels_a, labels_b, labels_c)?;
            let mutual_info = mutual_info_score::<F>(labels_a, labels_b)?;

            Ok(conditional_mi - mutual_info)
        }

        /// Calculate Information Quality Ratio
        pub fn information_quality_ratio<F>(
            true_labels: ArrayView1<i32>,
            predicted_labels: ArrayView1<i32>,
        ) -> Result<F>
        where
            F: Float + FromPrimitive + Debug,
        {
            let mi = mutual_info_score::<F>(true_labels, predicted_labels)?;
            let h_true = entropy::<F>(true_labels)?;
            let h_pred = entropy::<F>(predicted_labels)?;

            // IQR = 2 * MI / (H(true) + H(pred))
            let denominator = h_true + h_pred;
            if denominator == F::zero() {
                Ok(F::zero())
            } else {
                Ok(F::from(2).unwrap() * mi / denominator)
            }
        }

        /// Calculate Normalized Compression Distance for clustering similarity
        pub fn normalized_compression_distance<F>(
            labels_a: ArrayView1<i32>,
            labels_b: ArrayView1<i32>,
        ) -> Result<F>
        where
            F: Float + FromPrimitive + Debug,
        {
            // Convert labels to strings for compression
            let str_a = Self::labels_to_string(&labels_a);
            let str_b = Self::labels_to_string(&labels_b);
            let str_concat = format!("{}{}", str_a, str_b);

            // Simulate compression using simple string-based heuristic
            // In practice, you would use actual compression algorithms
            let len_a = Self::simulated_compression_length(&str_a);
            let len_b = Self::simulated_compression_length(&str_b);
            let len_concat = Self::simulated_compression_length(&str_concat);

            let numerator = len_concat as f64 - len_a.min(len_b) as f64;
            let denominator = len_a.max(len_b) as f64;

            if denominator == 0.0 {
                Ok(F::zero())
            } else {
                Ok(F::from(numerator / denominator).unwrap())
            }
        }

        /// Convert labels to string representation
        fn labels_to_string(labels: &ArrayView1<i32>) -> String {
            labels
                .iter()
                .map(|&x| x.to_string())
                .collect::<Vec<_>>()
                .join(",")
        }

        /// Simulate compression length using run-length encoding approximation
        fn simulated_compression_length(s: &str) -> usize {
            if s.is_empty() {
                return 0;
            }

            let mut compressed_length = 0;
            let chars: Vec<char> = s.chars().collect();
            let mut i = 0;

            while i < chars.len() {
                let current_char = chars[i];
                let mut run_length = 1;

                while i + run_length < chars.len() && chars[i + run_length] == current_char {
                    run_length += 1;
                }

                // Estimate compressed representation: character + run length
                compressed_length += if run_length > 1 {
                    2 + (run_length as f64).log10().ceil() as usize
                } else {
                    1
                };

                i += run_length;
            }

            compressed_length
        }

        /// Calculate Partition Coefficient for fuzzy clustering validation
        pub fn partition_coefficient<F>(_membershipmatrix: ArrayView2<F>) -> Result<F>
        where
            F: Float + FromPrimitive + Debug,
        {
            if _membershipmatrix.is_empty() {
                return Ok(F::zero());
            }

            let n_samples = _membershipmatrix.nrows();
            let mut sum_squares = F::zero();

            for i in 0..n_samples {
                for j in 0.._membershipmatrix.ncols() {
                    let membership = _membershipmatrix[[i, j]];
                    sum_squares = sum_squares + membership * membership;
                }
            }

            Ok(sum_squares / F::from(n_samples).unwrap())
        }

        /// Calculate Partition Entropy for fuzzy clustering validation
        pub fn partition_entropy<F>(_membershipmatrix: ArrayView2<F>) -> Result<F>
        where
            F: Float + FromPrimitive + Debug,
        {
            if _membershipmatrix.is_empty() {
                return Ok(F::zero());
            }

            let n_samples = _membershipmatrix.nrows();
            let mut entropy_sum = F::zero();

            for i in 0..n_samples {
                for j in 0.._membershipmatrix.ncols() {
                    let membership = _membershipmatrix[[i, j]];
                    if membership > F::zero() {
                        entropy_sum = entropy_sum - membership * membership.ln();
                    }
                }
            }

            Ok(entropy_sum / F::from(n_samples).unwrap())
        }

        /// Calculate Xie-Beni Index for fuzzy clustering validation
        pub fn xie_beni_index<F>(
            data: ArrayView2<F>,
            centers: ArrayView2<F>,
            _membershipmatrix: ArrayView2<F>,
            m: F, // Fuzzification parameter
        ) -> Result<F>
        where
            F: Float + FromPrimitive + Debug,
        {
            if data.nrows() != _membershipmatrix.nrows() {
                return Err(ClusteringError::InvalidInput(
                    "Data and membership _matrix must have same number of samples".to_string(),
                ));
            }

            if centers.nrows() != _membershipmatrix.ncols() {
                return Err(ClusteringError::InvalidInput(
                    "Centers and membership _matrix must be compatible".to_string(),
                ));
            }

            let n_samples = data.nrows();
            let n_clusters = centers.nrows();

            // Calculate within-cluster sum of squares
            let mut wcss = F::zero();
            for i in 0..n_samples {
                for k in 0..n_clusters {
                    let membership = _membershipmatrix[[i, k]];
                    let membership_powered = Self::power(membership, m);

                    // Calculate squared distance to center
                    let mut dist_sq = F::zero();
                    for j in 0..data.ncols() {
                        let diff = data[[i, j]] - centers[[k, j]];
                        dist_sq = dist_sq + diff * diff;
                    }

                    wcss = wcss + membership_powered * dist_sq;
                }
            }

            // Calculate minimum distance between centers
            let mut min_center_dist_sq = F::infinity();
            for i in 0..n_clusters {
                for j in (i + 1)..n_clusters {
                    let mut dist_sq = F::zero();
                    for k in 0..centers.ncols() {
                        let diff = centers[[i, k]] - centers[[j, k]];
                        dist_sq = dist_sq + diff * diff;
                    }
                    if dist_sq < min_center_dist_sq {
                        min_center_dist_sq = dist_sq;
                    }
                }
            }

            if min_center_dist_sq == F::zero() {
                return Ok(F::infinity());
            }

            Ok(wcss / (F::from(n_samples).unwrap() * min_center_dist_sq))
        }

        /// Helper function to compute power of a float (simple implementation)
        fn power<F: Float + FromPrimitive>(base: F, exponent: F) -> F {
            // Simple implementation using logarithms
            if base <= F::zero() {
                return F::zero();
            }
            (base.ln() * exponent).exp()
        }
    }

    /// Consensus clustering for ensemble validation
    pub struct ConsensusValidator;

    impl ConsensusValidator {
        /// Calculate consensus matrix from multiple clustering solutions
        pub fn _consensusmatrix(_clusteringsolutions: &[Array1<usize>]) -> Result<Array2<f64>> {
            if _clusteringsolutions.is_empty() {
                return Err(ClusteringError::InvalidInput(
                    "At least one clustering solution required".to_string(),
                ));
            }

            let n_samples = _clusteringsolutions[0].len();
            for solution in _clusteringsolutions {
                if solution.len() != n_samples {
                    return Err(ClusteringError::InvalidInput(
                        "All clustering _solutions must have the same number of samples"
                            .to_string(),
                    ));
                }
            }

            let mut consensus = Array2::zeros((n_samples, n_samples));
            let n_solutions = _clusteringsolutions.len() as f64;

            // For each pair of samples, count how often they are in the same cluster
            for solution in _clusteringsolutions {
                for i in 0..n_samples {
                    for j in 0..n_samples {
                        if solution[i] == solution[j] {
                            consensus[[i, j]] += 1.0;
                        }
                    }
                }
            }

            // Normalize by number of _solutions
            consensus.mapv_inplace(|x| x / n_solutions);

            Ok(consensus)
        }

        /// Calculate consensus clustering stability
        pub fn consensus_stability(_consensusmatrix: ArrayView2<f64>) -> Result<f64> {
            if _consensusmatrix.nrows() != _consensusmatrix.ncols() {
                return Err(ClusteringError::InvalidInput(
                    "Consensus _matrix must be square".to_string(),
                ));
            }

            let n_samples = _consensusmatrix.nrows();
            let mut stability_sum = 0.0;
            let mut count = 0;

            // Calculate stability as average consensus value
            for i in 0..n_samples {
                for j in (i + 1)..n_samples {
                    let consensus_value = _consensusmatrix[[i, j]];
                    // Stability is how close consensus is to 0 or 1 (binary decisions)
                    let stability = 1.0 - 4.0 * consensus_value * (1.0 - consensus_value);
                    stability_sum += stability;
                    count += 1;
                }
            }

            if count == 0 {
                Ok(1.0)
            } else {
                Ok(stability_sum / count as f64)
            }
        }

        /// Calculate Area Under CDF for consensus clustering
        pub fn area_under_cdf(_consensusmatrix: ArrayView2<f64>) -> Result<f64> {
            let n_samples = _consensusmatrix.nrows();
            let mut values = Vec::new();

            // Collect all consensus values (excluding diagonal)
            for i in 0..n_samples {
                for j in (i + 1)..n_samples {
                    values.push(_consensusmatrix[[i, j]]);
                }
            }

            if values.is_empty() {
                return Ok(0.0);
            }

            values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            // Calculate area under empirical CDF
            let mut area = 0.0;
            let n_values = values.len() as f64;

            for (i, &value) in values.iter().enumerate() {
                let cdf_value = (i + 1) as f64 / n_values;
                if i == 0 {
                    area += value * cdf_value;
                } else {
                    let prev_value = values[i - 1];
                    area += (value - prev_value) * cdf_value;
                }
            }

            Ok(area)
        }

        /// Determine optimal number of clusters using consensus analysis
        pub fn optimal_k_consensus(
            consensus_matrices: &[Array2<f64>],
            k_values: &[usize],
        ) -> Result<usize> {
            if consensus_matrices.len() != k_values.len() {
                return Err(ClusteringError::InvalidInput(
                    "Number of consensus _matrices must match number of k _values".to_string(),
                ));
            }

            let mut best_k = k_values[0];
            let mut best_score = 0.0;

            for (i, &k) in k_values.iter().enumerate() {
                let _consensusmatrix = &consensus_matrices[i];
                let stability = Self::consensus_stability(_consensusmatrix.view())?;
                let auc = Self::area_under_cdf(_consensusmatrix.view())?;

                // Combined score: high stability and high AUC indicate good clustering
                let score = stability * auc;

                if score > best_score {
                    best_score = score;
                    best_k = k;
                }
            }

            Ok(best_k)
        }
    }

    /// Information-theoretic feature selection for clustering
    pub struct InformationFeatureSelector<F: Float + FromPrimitive + Debug> {
        /// Feature relevance scores
        relevance_scores: Vec<F>,
        /// Feature redundancy matrix
        redundancy_matrix: Array2<F>,
        /// Selected feature indices
        selectedfeatures: Vec<usize>,
    }

    impl<F: Float + FromPrimitive + Debug> InformationFeatureSelector<F> {
        /// Create new feature selector
        pub fn new(_nfeatures: usize) -> Self {
            Self {
                relevance_scores: vec![F::zero(); _nfeatures],
                redundancy_matrix: Array2::zeros((_nfeatures, _nfeatures)),
                selectedfeatures: Vec::new(),
            }
        }

        /// Select features using mutual information maximization
        pub fn selectfeatures(
            &mut self,
            data: ArrayView2<F>,
            target_labels: ArrayView1<i32>,
            nfeatures_to_select: usize,
        ) -> Result<Vec<usize>> {
            let nfeatures = data.ncols();

            // Calculate relevance scores (MI with target)
            for j in 0..nfeatures {
                let featuredata = data.column(j);
                let discretizedfeature = self.discretizefeature(featuredata)?;
                let relevance = mutual_info_score::<F>(target_labels, discretizedfeature.view())?;
                self.relevance_scores[j] = relevance;
            }

            // Calculate redundancy matrix (MI between features)
            for i in 0..nfeatures {
                for j in (i + 1)..nfeatures {
                    let feature_i = self.discretizefeature(data.column(i))?;
                    let feature_j = self.discretizefeature(data.column(j))?;
                    let redundancy = mutual_info_score::<F>(feature_i.view(), feature_j.view())?;
                    self.redundancy_matrix[[i, j]] = redundancy;
                    self.redundancy_matrix[[j, i]] = redundancy;
                }
            }

            // MRMR (Maximum Relevance Minimum Redundancy) selection
            self.selectedfeatures.clear();
            let mut availablefeatures: Vec<usize> = (0..nfeatures).collect();

            // Select first feature with highest relevance
            if let Some((idx, &max_relevance_idx)) =
                availablefeatures
                    .iter()
                    .enumerate()
                    .max_by(|(_, &a), (_, &b)| {
                        self.relevance_scores[a]
                            .partial_cmp(&self.relevance_scores[b])
                            .unwrap()
                    })
            {
                self.selectedfeatures.push(max_relevance_idx);
                availablefeatures.remove(idx);
            }

            // Select remaining features using MRMR criterion
            while self.selectedfeatures.len() < nfeatures_to_select && !availablefeatures.is_empty()
            {
                let mut bestfeature = availablefeatures[0];
                let mut best_score = F::neg_infinity();

                for &candidate in &availablefeatures {
                    let relevance = self.relevance_scores[candidate];

                    // Calculate average redundancy with already selected features
                    let mut redundancy_sum = F::zero();
                    for &selected in &self.selectedfeatures {
                        redundancy_sum =
                            redundancy_sum + self.redundancy_matrix[[candidate, selected]];
                    }
                    let avg_redundancy = if self.selectedfeatures.is_empty() {
                        F::zero()
                    } else {
                        redundancy_sum / F::from(self.selectedfeatures.len()).unwrap()
                    };

                    // MRMR score: relevance - redundancy
                    let mrmr_score = relevance - avg_redundancy;

                    if mrmr_score > best_score {
                        best_score = mrmr_score;
                        bestfeature = candidate;
                    }
                }

                self.selectedfeatures.push(bestfeature);
                availablefeatures.retain(|&x| x != bestfeature);
            }

            Ok(self.selectedfeatures.clone())
        }

        /// Discretize continuous features for mutual information calculation
        fn discretizefeature(&self, feature: ArrayView1<F>) -> Result<Array1<i32>> {
            if feature.is_empty() {
                return Ok(Array1::zeros(0));
            }

            // Find min and max values
            let mut min_val = feature[0];
            let mut max_val = feature[0];
            for &val in feature.iter() {
                if val < min_val {
                    min_val = val;
                }
                if val > max_val {
                    max_val = val;
                }
            }

            let range = max_val - min_val;
            if range == F::zero() {
                // All values are the same
                return Ok(Array1::zeros(feature.len()));
            }

            // Use equal-width discretization with 10 bins
            let n_bins = 10;
            let bin_width = range / F::from(n_bins).unwrap();

            let discretized = feature
                .iter()
                .map(|&val| {
                    let bin = ((val - min_val) / bin_width).to_usize().unwrap_or(0);
                    bin.min(n_bins - 1) as i32
                })
                .collect();

            Ok(Array1::from_vec(discretized))
        }

        /// Get feature importance scores
        pub fn getfeature_scores(&self) -> &[F] {
            &self.relevance_scores
        }

        /// Get selected feature indices
        pub fn get_selectedfeatures(&self) -> &[usize] {
            &self.selectedfeatures
        }
    }
}

/// Advanced information-theoretic clustering metrics
pub mod advanced_metrics {
    use super::*;
    use std::collections::BTreeMap;

    /// Calculate Jensen-Shannon divergence for clustering evaluation
    ///
    /// Jensen-Shannon divergence is a symmetrized and smoothed version of KL divergence
    /// that measures the similarity between probability distributions. For clustering,
    /// it can be used to compare the distribution of clusters between different methods.
    ///
    /// # Arguments
    ///
    /// * `labels_true` - Ground truth class labels
    /// * `labelspred` - Predicted cluster labels
    ///
    /// # Returns
    ///
    /// The Jensen-Shannon divergence (0 to 1, lower is better)
    pub fn jensen_shannon_divergence<F>(
        labels_true: ArrayView1<i32>,
        labelspred: ArrayView1<i32>,
    ) -> Result<F>
    where
        F: Float + FromPrimitive + Debug,
    {
        if labels_true.len() != labelspred.len() {
            return Err(ClusteringError::InvalidInput(
                "True and predicted labels must have the same length".to_string(),
            ));
        }

        // Convert to probability distributions
        let p = cluster_distribution(labels_true)?;
        let q = cluster_distribution(labelspred)?;

        // Align distributions (handle different cluster numbers)
        let (p_aligned, q_aligned) = align_distributions(p, q);

        // Calculate Jensen-Shannon divergence
        let js_div = jensen_shannon_divergence_core(p_aligned, q_aligned)?;
        Ok(js_div)
    }

    /// Convert cluster labels to probability distribution
    fn cluster_distribution<F>(labels: ArrayView1<i32>) -> Result<BTreeMap<i32, F>>
    where
        F: Float + FromPrimitive,
    {
        let mut counts: BTreeMap<i32, usize> = BTreeMap::new();
        let total = labels.len();

        for &label in labels.iter() {
            if label >= 0 {
                // Ignore noise points (negative labels)
                *counts.entry(label).or_insert(0) += 1;
            }
        }

        let mut distribution = BTreeMap::new();
        for (label, count) in counts {
            let prob = F::from(count).unwrap() / F::from(total).unwrap();
            distribution.insert(label, prob);
        }

        Ok(distribution)
    }

    /// Align two probability distributions for comparison
    fn align_distributions<F>(mut p: BTreeMap<i32, F>, mut q: BTreeMap<i32, F>) -> (Vec<F>, Vec<F>)
    where
        F: Float + Copy,
    {
        // Get all unique labels
        let mut all_labels: Vec<i32> = p.keys().chain(q.keys()).copied().collect();
        all_labels.sort_unstable();
        all_labels.dedup();

        let mut p_aligned = Vec::new();
        let mut q_aligned = Vec::new();

        for label in all_labels {
            p_aligned.push(p.remove(&label).unwrap_or(F::zero()));
            q_aligned.push(q.remove(&label).unwrap_or(F::zero()));
        }

        (p_aligned, q_aligned)
    }

    /// Core Jensen-Shannon divergence calculation
    fn jensen_shannon_divergence_core<F>(p: Vec<F>, q: Vec<F>) -> Result<F>
    where
        F: Float + FromPrimitive,
    {
        if p.len() != q.len() {
            return Err(ClusteringError::InvalidInput(
                "Distributions must have same length".to_string(),
            ));
        }

        let mut js_div = F::zero();
        let half = F::from(0.5).unwrap();

        for (p_i, q_i) in p.iter().zip(q.iter()) {
            let m_i = half * (*p_i + *q_i);

            if *p_i > F::zero() && m_i > F::zero() {
                js_div = js_div + half * *p_i * (*p_i / m_i).ln();
            }

            if *q_i > F::zero() && m_i > F::zero() {
                js_div = js_div + half * *q_i * (*q_i / m_i).ln();
            }
        }

        Ok(js_div)
    }

    /// Calculate clustering stability using bootstrap resampling
    ///
    /// This metric measures how stable a clustering algorithm is by running it
    /// on multiple bootstrap samples and measuring the consistency of results.
    ///
    /// # Arguments
    ///
    /// * `data` - Input data matrix
    /// * `clustering_fn` - Function that takes data and returns cluster labels
    /// * `n_bootstrap` - Number of bootstrap samples to generate
    /// * `sample_ratio` - Fraction of data to sample in each bootstrap
    ///
    /// # Returns
    ///
    /// Stability score (0 to 1, higher is better)
    pub fn clustering_stability<F, ClusterFn>(
        data: &Array2<F>,
        clustering_fn: ClusterFn,
        n_bootstrap: usize,
        sample_ratio: f64,
    ) -> Result<F>
    where
        F: Float + FromPrimitive + Debug + Send + Sync,
        ClusterFn: Fn(&Array2<F>) -> Result<Array1<i32>> + Sync,
    {
        let n_samples = data.nrows();
        let sample_size = (n_samples as f64 * sample_ratio) as usize;

        if sample_size < 2 {
            return Err(ClusteringError::InvalidInput(
                "Sample size too small for stability analysis".to_string(),
            ));
        }

        // Generate _bootstrap samples and cluster them
        let mut all_labels = Vec::new();
        let mut rng = rand::rng();

        for _ in 0..n_bootstrap {
            // Bootstrap sampling
            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(&mut rng);
            indices.truncate(sample_size);
            indices.sort_unstable();

            // Create _bootstrap sample
            let mut bootstrapdata = Array2::zeros((sample_size, data.ncols()));
            for (i, &idx) in indices.iter().enumerate() {
                bootstrapdata.row_mut(i).assign(&data.row(idx));
            }

            // Apply clustering
            let labels = clustering_fn(&bootstrapdata)?;
            all_labels.push((indices, labels));
        }

        // Calculate pairwise stability
        let mut total_stability = F::zero();
        let mut n_comparisons = 0;

        for i in 0..all_labels.len() {
            for j in (i + 1)..all_labels.len() {
                let stability = compute_pairwise_stability(&all_labels[i], &all_labels[j])?;
                total_stability = total_stability + stability;
                n_comparisons += 1;
            }
        }

        if n_comparisons > 0 {
            Ok(total_stability / F::from(n_comparisons).unwrap())
        } else {
            Ok(F::zero())
        }
    }

    /// Compute stability between two bootstrap clustering results
    fn compute_pairwise_stability<F>(
        result1: &(Vec<usize>, Array1<i32>),
        result2: &(Vec<usize>, Array1<i32>),
    ) -> Result<F>
    where
        F: Float + FromPrimitive + Debug,
    {
        let (indices1, labels1) = result1;
        let (indices2, labels2) = result2;

        // Find common samples
        let mut common_samples = Vec::new();
        let mut map1 = HashMap::new();
        let mut map2 = HashMap::new();

        for (pos, &idx) in indices1.iter().enumerate() {
            map1.insert(idx, pos);
        }

        for (pos, &idx) in indices2.iter().enumerate() {
            if let Some(&pos1) = map1.get(&idx) {
                map2.insert(idx, pos);
                common_samples.push((pos1, pos, idx));
            }
        }

        if common_samples.len() < 2 {
            return Ok(F::zero());
        }

        // Extract labels for common samples
        let mut labels1_common = Array1::zeros(common_samples.len());
        let mut labels2_common = Array1::zeros(common_samples.len());

        for (i, &(pos1, pos2, _)) in common_samples.iter().enumerate() {
            labels1_common[i] = labels1[pos1];
            labels2_common[i] = labels2[pos2];
        }

        // Calculate adjusted mutual information as stability measure
        adjusted_mutual_info_score(labels1_common.view(), labels2_common.view())
    }

    /// Calculate information gain for feature selection in clustering
    ///
    /// Information gain measures how much information a feature provides
    /// about the cluster structure.
    ///
    /// # Arguments
    ///
    /// * `data` - Input data matrix (samples x features)
    /// * `labels` - Cluster labels
    /// * `feature_idx` - Index of the feature to evaluate
    ///
    /// # Returns
    ///
    /// Information gain score (higher is better)
    pub fn information_gain<F>(
        data: &Array2<F>,
        labels: ArrayView1<i32>,
        feature_idx: usize,
    ) -> Result<F>
    where
        F: Float + FromPrimitive + Debug,
    {
        if feature_idx >= data.ncols() {
            return Err(ClusteringError::InvalidInput(
                "Feature index out of bounds".to_string(),
            ));
        }

        if data.nrows() != labels.len() {
            return Err(ClusteringError::InvalidInput(
                "Data and labels must have same number of samples".to_string(),
            ));
        }

        // Calculate entropy of cluster labels
        let h_labels = entropy::<F>(labels)?;

        // Discretize the feature
        let feature_column = data.column(feature_idx);
        let discretizedfeature = discretizefeature(feature_column)?;

        // Calculate conditional entropy H(Labels | Feature)
        let h_conditional = conditional_entropy::<F>(labels, discretizedfeature.view())?;

        // Information gain = H(Labels) - H(Labels | Feature)
        Ok(h_labels - h_conditional)
    }

    /// Discretize a continuous feature into bins
    fn discretizefeature<F>(feature: ndarray::ArrayView1<F>) -> Result<Array1<i32>>
    where
        F: Float + FromPrimitive,
    {
        if feature.is_empty() {
            return Ok(Array1::zeros(0));
        }

        let mut values: Vec<F> = feature.to_vec();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min_val = values[0];
        let max_val = values[values.len() - 1];
        let range = max_val - min_val;

        if range == F::zero() {
            return Ok(Array1::zeros(feature.len()));
        }

        let n_bins = 10;
        let bin_width = range / F::from(n_bins).unwrap();

        let discretized: Vec<i32> = feature
            .iter()
            .map(|&val| {
                let bin = ((val - min_val) / bin_width).to_usize().unwrap_or(0);
                bin.min(n_bins - 1) as i32
            })
            .collect();

        Ok(Array1::from_vec(discretized))
    }

    /// Calculate conditional entropy H(Y | X)
    fn conditional_entropy<F>(y: ArrayView1<i32>, x: ArrayView1<i32>) -> Result<F>
    where
        F: Float + FromPrimitive + Debug,
    {
        if y.len() != x.len() {
            return Err(ClusteringError::InvalidInput(
                "Arrays must have same length".to_string(),
            ));
        }

        let n_samples = y.len();
        let mut joint_counts: HashMap<(i32, i32), usize> = HashMap::new();
        let mut x_counts: HashMap<i32, usize> = HashMap::new();

        // Count occurrences
        for (&y_val, &x_val) in y.iter().zip(x.iter()) {
            *joint_counts.entry((x_val, y_val)).or_insert(0) += 1;
            *x_counts.entry(x_val).or_insert(0) += 1;
        }

        let mut h_conditional = F::zero();
        let n_samples_f = F::from(n_samples).unwrap();

        // Calculate H(Y | X) = sumx P(x) * H(Y | X=x)
        for (&x_val, &x_count) in &x_counts {
            let px = F::from(x_count).unwrap() / n_samples_f;

            // Calculate H(Y | X=x)
            let mut hy_givenx = F::zero();
            let x_count_f = F::from(x_count).unwrap();

            for (&(x_jointy_val, _y_val), &joint_count) in &joint_counts {
                if x_jointy_val == x_val {
                    let py_givenx = F::from(joint_count).unwrap() / x_count_f;
                    if py_givenx > F::zero() {
                        hy_givenx = hy_givenx - py_givenx * py_givenx.ln();
                    }
                }
            }

            h_conditional = h_conditional + px * hy_givenx;
        }

        Ok(h_conditional)
    }

    /// Calculate the information cluster quality metric
    ///
    /// This metric combines multiple information-theoretic measures to provide
    /// a comprehensive assessment of clustering quality.
    ///
    /// # Arguments
    ///
    /// * `data` - Input data matrix
    /// * `labels` - Cluster labels
    ///
    /// # Returns
    ///
    /// Information cluster quality score (higher is better)
    pub fn information_cluster_quality<F>(data: &Array2<F>, labels: ArrayView1<i32>) -> Result<F>
    where
        F: Float + FromPrimitive + Debug,
    {
        if data.nrows() != labels.len() {
            return Err(ClusteringError::InvalidInput(
                "Data and labels must have same number of samples".to_string(),
            ));
        }

        // Calculate cluster entropy (lower is better for fewer, well-separated clusters)
        let h_clusters = entropy(labels)?;

        // Calculate average information gain across all features
        let mut total_info_gain = F::zero();
        let nfeatures = data.ncols();

        for feature_idx in 0..nfeatures {
            let info_gain = information_gain(data, labels, feature_idx)?;
            total_info_gain = total_info_gain + info_gain;
        }

        let avg_info_gain = if nfeatures > 0 {
            total_info_gain / F::from(nfeatures).unwrap()
        } else {
            F::zero()
        };

        // Calculate within-cluster information content
        let within_cluster_info = calculate_within_cluster_information(data, labels)?;

        // Combine metrics (this is a heuristic combination)
        let quality = avg_info_gain / (F::one() + h_clusters) * within_cluster_info;

        Ok(quality)
    }

    /// Calculate within-cluster information content
    fn calculate_within_cluster_information<F>(
        data: &Array2<F>,
        labels: ArrayView1<i32>,
    ) -> Result<F>
    where
        F: Float + FromPrimitive + Debug,
    {
        let mut clusterdata: HashMap<i32, Vec<usize>> = HashMap::new();

        // Group samples by cluster
        for (i, &label) in labels.iter().enumerate() {
            if label >= 0 {
                // Ignore noise points
                clusterdata.entry(label).or_insert_with(Vec::new).push(i);
            }
        }

        let mut total_info = F::zero();
        let mut total_samples = 0;

        // Calculate information content within each cluster
        for (_, sample_indices) in clusterdata {
            if sample_indices.len() > 1 {
                let cluster_size = sample_indices.len();

                // Calculate variance within cluster (as a proxy for information)
                let mut cluster_variance = F::zero();

                for feature_idx in 0..data.ncols() {
                    let feature_values: Vec<F> = sample_indices
                        .iter()
                        .map(|&i| data[[i, feature_idx]])
                        .collect();

                    let mean = feature_values.iter().fold(F::zero(), |acc, &x| acc + x)
                        / F::from(cluster_size).unwrap();
                    let variance = feature_values
                        .iter()
                        .map(|&x| (x - mean) * (x - mean))
                        .fold(F::zero(), |acc, x| acc + x)
                        / F::from(cluster_size).unwrap();

                    cluster_variance = cluster_variance + variance;
                }

                // Information content is inversely related to variance
                let cluster_info = F::one() / (F::one() + cluster_variance);
                total_info = total_info + cluster_info * F::from(cluster_size).unwrap();
                total_samples += cluster_size;
            }
        }

        if total_samples > 0 {
            Ok(total_info / F::from(total_samples).unwrap())
        } else {
            Ok(F::zero())
        }
    }
}
