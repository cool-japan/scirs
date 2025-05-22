//! Adjusted Rand Index (ARI) implementation
//!
//! The Adjusted Rand Index is a metric for measuring the similarity between two clusterings,
//! adjusted for the chance of random agreements.

use ndarray::{ArrayBase, Data, Dimension};
use std::collections::HashMap;
use std::fmt::Debug;

use crate::error::{MetricsError, Result};

/// Calculates the Adjusted Rand Index (ARI) between two clusterings
///
/// The Adjusted Rand Index is a measure of the similarity between two clusterings,
/// adjusted for the chance of random agreements. It ranges from -1 to 1, with:
/// - 1: Perfect agreement
/// - 0: Agreement equivalent to random chance
/// - Negative values: Agreement worse than random chance
///
/// # Arguments
///
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Predicted cluster labels to evaluate
///
/// # Returns
///
/// * The Adjusted Rand Index (between -1.0 and 1.0)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::clustering::external_metrics::adjusted_rand_index;
///
/// let labels_true = array![0, 0, 1, 1, 2, 2];
/// let labels_pred = array![0, 0, 0, 1, 1, 1];
///
/// let ari = adjusted_rand_index(&labels_true, &labels_pred).unwrap();
/// ```
pub fn adjusted_rand_index<T, U, S1, S2, D1, D2>(
    labels_true: &ArrayBase<S1, D1>,
    labels_pred: &ArrayBase<S2, D2>,
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

    // Compute contingency matrix
    let mut contingency: HashMap<(String, String), usize> = HashMap::new();
    for (lt, lp) in labels_true.iter().zip(labels_pred.iter()) {
        let key = (format!("{:?}", lt), format!("{:?}", lp));
        *contingency.entry(key).or_insert(0) += 1;
    }

    // Count pairs
    let (mut sum_comb_a, mut sum_comb_b, mut sum_comb) = (0.0, 0.0, 0.0);

    // Count terms for true labels
    let mut a_counts: HashMap<String, usize> = HashMap::new();
    for lt in labels_true.iter() {
        let key = format!("{:?}", lt);
        *a_counts.entry(key).or_insert(0) += 1;
    }

    for (_, &count) in a_counts.iter() {
        if count > 1 {
            sum_comb_a += combinations(count);
        }
    }

    // Count terms for predicted labels
    let mut b_counts: HashMap<String, usize> = HashMap::new();
    for lp in labels_pred.iter() {
        let key = format!("{:?}", lp);
        *b_counts.entry(key).or_insert(0) += 1;
    }

    for (_, &count) in b_counts.iter() {
        if count > 1 {
            sum_comb_b += combinations(count);
        }
    }

    // Count terms for contingency matrix
    for (_, &count) in contingency.iter() {
        if count > 1 {
            sum_comb += combinations(count);
        }
    }

    // Calculate total number of pairs
    let n_pairs = combinations(n_samples);

    // Calculate ARI
    let expected_index = sum_comb_a * sum_comb_b / n_pairs;
    let max_index = (sum_comb_a + sum_comb_b) / 2.0;

    if max_index == expected_index {
        // Special case: perfect match
        Ok(1.0)
    } else {
        Ok((sum_comb - expected_index) / (max_index - expected_index))
    }
}

/// Helper function to calculate nC2 (number of ways to choose 2 items from n)
fn combinations(n: usize) -> f64 {
    if n < 2 {
        0.0
    } else {
        (n * (n - 1)) as f64 / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_adjusted_rand_index() {
        // Perfect match
        let labels_true = array![0, 0, 1, 1, 2, 2];
        let labels_pred = array![1, 1, 0, 0, 2, 2]; // Same clustering with different labels
        let ari = adjusted_rand_index(&labels_true, &labels_pred).unwrap();
        assert!(
            (ari - 1.0).abs() < 1e-10,
            "ARI should be 1.0 for perfect match"
        );

        // Imperfect match (different cluster arrangement)
        let labels_true = array![0, 0, 0, 1, 1, 1];
        let labels_pred = array![0, 1, 2, 0, 1, 2];
        let ari = adjusted_rand_index(&labels_true, &labels_pred).unwrap();
        assert!(
            (-1.0..=1.0).contains(&ari),
            "ARI should be between -1 and 1, got: {}",
            ari
        );

        // Partial match
        let labels_true = array![0, 0, 1, 1, 2, 2];
        let labels_pred = array![0, 0, 0, 1, 1, 1];
        let ari = adjusted_rand_index(&labels_true, &labels_pred).unwrap();
        assert!(
            ari > 0.0 && ari < 1.0,
            "ARI should be between 0 and 1 for partial match"
        );
    }
}
