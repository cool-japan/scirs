//! Advanced information-theoretic clustering metrics
//!
//! This module provides advanced information-theoretic metrics for clustering
//! evaluation, including Jensen-Shannon divergence, variation of information,
//! and other specialized measures.

use scirs2_core::ndarray::ArrayView1;
use scirs2_core::numeric::{Float, FromPrimitive};
use std::collections::{BTreeMap, HashMap};
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};

/// Calculate Jensen-Shannon divergence between two clusterings
///
/// Jensen-Shannon divergence is a symmetric and bounded measure of similarity
/// between two probability distributions. For clustering, it measures the
/// divergence between the cluster size distributions.
///
/// # Arguments
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Predicted cluster labels
///
/// # Returns
/// * `Result<F>` - The Jensen-Shannon divergence (0 to 1, lower is better)
///
/// # Example
/// ```rust
/// use scirs2_core::ndarray::Array1;
/// use scirs2_cluster::metrics::jensen_shannon_divergence;
///
/// let true_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
/// let pred_labels = Array1::from_vec(vec![0, 0, 1, 1, 1, 2]);
///
/// let js: f64 = jensen_shannon_divergence(true_labels.view(), pred_labels.view()).unwrap();
/// assert!(js >= 0.0 && js <= 1.0);
/// ```
pub fn jensen_shannon_divergence<F>(
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

    // Convert to probability distributions
    let p = cluster_distribution(labels_true)?;
    let q = cluster_distribution(labels_pred)?;

    // Align distributions (handle different cluster numbers)
    let (p_aligned, q_aligned) = align_distributions(p, q);

    // Calculate Jensen-Shannon divergence
    jensen_shannon_divergence_core(p_aligned, q_aligned)
}

/// Calculate normalized variation of information between two clusterings
///
/// Variation of Information (VI) is the sum of conditional entropies H(X|Y) + H(Y|X).
/// The normalized version scales it to the range [0, 1] for easier interpretation.
///
/// # Arguments
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Predicted cluster labels
///
/// # Returns
/// * `Result<F>` - The normalized variation of information (0 to 1, lower is better)
pub fn normalized_variation_of_information<F>(
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

    // Calculate individual entropies
    let h_true = entropy::<F>(labels_true)?;
    let h_pred = entropy::<F>(labels_pred)?;

    // Calculate conditional entropies
    let h_true_given_pred = conditional_entropy::<F>(labels_true, labels_pred)?;
    let h_pred_given_true = conditional_entropy::<F>(labels_pred, labels_true)?;

    // Variation of information
    let vi = h_true_given_pred + h_pred_given_true;

    // Normalize by the maximum possible VI (log(n))
    let max_vi = F::from(n_samples as f64).unwrap().ln();

    if max_vi == F::zero() {
        Ok(F::zero())
    } else {
        Ok(vi / max_vi)
    }
}

/// Calculate the symmetric information coefficient
///
/// This is a normalized measure of association between two clusterings based
/// on their mutual information, similar to the Pearson correlation coefficient.
///
/// # Arguments
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Predicted cluster labels
///
/// # Returns
/// * `Result<F>` - The symmetric information coefficient (-1 to 1, higher is better)
pub fn symmetric_information_coefficient<F>(
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

    let mi = mutual_information::<F>(labels_true, labels_pred)?;
    let h_true = entropy::<F>(labels_true)?;
    let h_pred = entropy::<F>(labels_pred)?;

    // Handle degenerate cases
    if h_true == F::zero() && h_pred == F::zero() {
        return Ok(F::one());
    }

    let joint_entropy = h_true + h_pred - mi;
    if joint_entropy == F::zero() {
        Ok(F::one())
    } else {
        Ok(mi / joint_entropy)
    }
}

/// Calculate information gain ratio between clusterings
///
/// Information gain ratio is the ratio of mutual information to the entropy
/// of the true clustering. It measures how much information the predicted
/// clustering provides about the true clustering.
///
/// # Arguments
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Predicted cluster labels
///
/// # Returns
/// * `Result<F>` - The information gain ratio (0 to 1, higher is better)
pub fn information_gain_ratio<F>(
    labels_true: ArrayView1<i32>,
    labels_pred: ArrayView1<i32>,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let mi = mutual_information::<F>(labels_true, labels_pred)?;
    let h_true = entropy::<F>(labels_true)?;

    if h_true == F::zero() {
        Ok(F::one())
    } else {
        Ok(mi / h_true)
    }
}

/// Calculate uncertainty coefficient (also known as proficiency)
///
/// This measures the proportional reduction in uncertainty about one clustering
/// when the other clustering is known. It's asymmetric: U(X|Y) ` U(Y|X).
///
/// # Arguments
/// * `labels_x` - First clustering labels
/// * `labels_y` - Second clustering labels
///
/// # Returns
/// * `Result<F>` - The uncertainty coefficient (0 to 1, higher is better)
pub fn uncertainty_coefficient<F>(labels_x: ArrayView1<i32>, labels_y: ArrayView1<i32>) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let mi = mutual_information::<F>(labels_x, labels_y)?;
    let h_x = entropy::<F>(labels_x)?;

    if h_x == F::zero() {
        Ok(F::one())
    } else {
        Ok(mi / h_x)
    }
}

/// Calculate symmetric uncertainty coefficient
///
/// This is the symmetric version of the uncertainty coefficient, calculated
/// as the geometric mean of U(X|Y) and U(Y|X).
///
/// # Arguments
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Predicted cluster labels
///
/// # Returns
/// * `Result<F>` - The symmetric uncertainty coefficient (0 to 1, higher is better)
pub fn symmetric_uncertainty_coefficient<F>(
    labels_true: ArrayView1<i32>,
    labels_pred: ArrayView1<i32>,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let u_xy = uncertainty_coefficient::<F>(labels_true, labels_pred)?;
    let u_yx = uncertainty_coefficient::<F>(labels_pred, labels_true)?;

    Ok((u_xy * u_yx).sqrt())
}

/// Convert cluster labels to probability distribution
fn cluster_distribution<F>(labels: ArrayView1<i32>) -> Result<BTreeMap<i32, F>>
where
    F: Float + FromPrimitive,
{
    let mut counts: BTreeMap<i32, usize> = BTreeMap::new();
    for &label in labels.iter() {
        *counts.entry(label).or_insert(0) += 1;
    }

    let total = labels.len();
    if total == 0 {
        return Ok(BTreeMap::new());
    }

    let total_f = F::from(total).unwrap();
    let mut distribution = BTreeMap::new();

    for (label, count) in counts {
        distribution.insert(label, F::from(count).unwrap() / total_f);
    }

    Ok(distribution)
}

/// Align two probability distributions to have the same support
fn align_distributions<F>(mut p: BTreeMap<i32, F>, mut q: BTreeMap<i32, F>) -> (Vec<F>, Vec<F>)
where
    F: Float + FromPrimitive,
{
    // Get all unique labels from both distributions
    let mut all_labels = std::collections::HashSet::new();
    for &label in p.keys() {
        all_labels.insert(label);
    }
    for &label in q.keys() {
        all_labels.insert(label);
    }

    let mut sorted_labels: Vec<_> = all_labels.into_iter().collect();
    sorted_labels.sort();

    let mut p_aligned = Vec::new();
    let mut q_aligned = Vec::new();

    for label in sorted_labels {
        p_aligned.push(*p.get(&label).unwrap_or(&F::zero()));
        q_aligned.push(*q.get(&label).unwrap_or(&F::zero()));
    }

    (p_aligned, q_aligned)
}

/// Core Jensen-Shannon divergence calculation
fn jensen_shannon_divergence_core<F>(p: Vec<F>, q: Vec<F>) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    if p.len() != q.len() {
        return Err(ClusteringError::InvalidInput(
            "Probability distributions must have the same length".to_string(),
        ));
    }

    let two = F::from(2).unwrap();

    // Calculate the average distribution M
    let m: Vec<F> = p
        .iter()
        .zip(q.iter())
        .map(|(&p_i, &q_i)| (p_i + q_i) / two)
        .collect();

    // Calculate KL(P || M)
    let mut kl_pm = F::zero();
    for (i, &p_i) in p.iter().enumerate() {
        if p_i > F::zero() && m[i] > F::zero() {
            kl_pm = kl_pm + p_i * (p_i / m[i]).ln();
        }
    }

    // Calculate KL(Q || M)
    let mut kl_qm = F::zero();
    for (i, &q_i) in q.iter().enumerate() {
        if q_i > F::zero() && m[i] > F::zero() {
            kl_qm = kl_qm + q_i * (q_i / m[i]).ln();
        }
    }

    // Jensen-Shannon divergence
    let js_div = (kl_pm + kl_qm) / two;

    // Return the Jensen-Shannon distance (square root of divergence)
    Ok(js_div.sqrt())
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

/// Calculate mutual information between two label assignments
fn mutual_information<F>(labels_true: ArrayView1<i32>, labels_pred: ArrayView1<i32>) -> Result<F>
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
    let mut contingency = HashMap::new();
    for (&true_label, &pred_label) in labels_true.iter().zip(labels_pred.iter()) {
        *contingency.entry((true_label, pred_label)).or_insert(0) += 1;
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_jensen_shannon_divergence_identical() {
        let labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        let js: f64 = jensen_shannon_divergence(labels.view(), labels.view()).unwrap();
        assert!(js.abs() < 1e-10); // Should be zero for identical distributions
    }

    #[test]
    fn test_normalized_variation_of_information() {
        let true_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        let pred_labels = Array1::from_vec(vec![0, 0, 1, 1, 1, 2]);

        let nvi: f64 =
            normalized_variation_of_information(true_labels.view(), pred_labels.view()).unwrap();
        assert!(nvi >= 0.0 && nvi <= 1.0);
    }

    #[test]
    fn test_symmetric_information_coefficient() {
        let true_labels = Array1::from_vec(vec![0, 0, 1, 1]);
        let pred_labels = Array1::from_vec(vec![0, 1, 0, 1]);

        let sic: f64 =
            symmetric_information_coefficient(true_labels.view(), pred_labels.view()).unwrap();
        assert!(sic >= -1.0 && sic <= 1.0);
    }

    #[test]
    fn test_information_gain_ratio() {
        let true_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        let pred_labels = Array1::from_vec(vec![0, 0, 1, 1, 1, 2]);

        let igr: f64 = information_gain_ratio(true_labels.view(), pred_labels.view()).unwrap();
        assert!(igr >= 0.0 && igr <= 1.0);
    }

    #[test]
    fn test_uncertainty_coefficient() {
        let x = Array1::from_vec(vec![0, 0, 1, 1]);
        let y = Array1::from_vec(vec![0, 1, 0, 1]);

        let uc: f64 = uncertainty_coefficient::<f64>(x.view(), y.view()).unwrap();
        assert!(uc >= 0.0 && uc <= 1.0);
    }

    #[test]
    fn test_symmetric_uncertainty_coefficient() {
        let true_labels = Array1::from_vec(vec![0, 0, 1, 1]);
        let pred_labels = Array1::from_vec(vec![0, 1, 0, 1]);

        let suc: f64 =
            symmetric_uncertainty_coefficient::<f64>(true_labels.view(), pred_labels.view())
                .unwrap();
        assert!(suc >= 0.0 && suc <= 1.0);
    }

    #[test]
    fn test_cluster_distribution() {
        let labels = Array1::from_vec(vec![0, 0, 1, 1, 2]);
        let dist = cluster_distribution::<f64>(labels.view()).unwrap();

        assert_eq!(dist.len(), 3);
        assert!((dist[&0] - 0.4).abs() < 1e-10);
        assert!((dist[&1] - 0.4).abs() < 1e-10);
        assert!((dist[&2] - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_align_distributions() {
        let mut p = BTreeMap::new();
        p.insert(0, 0.5);
        p.insert(1, 0.5);

        let mut q = BTreeMap::new();
        q.insert(1, 0.3);
        q.insert(2, 0.7);

        let (p_aligned, q_aligned) = align_distributions(p, q);

        assert_eq!(p_aligned.len(), 3);
        assert_eq!(q_aligned.len(), 3);
        assert_eq!(p_aligned, vec![0.5, 0.5, 0.0]);
        assert_eq!(q_aligned, vec![0.0, 0.3, 0.7]);
    }
}
