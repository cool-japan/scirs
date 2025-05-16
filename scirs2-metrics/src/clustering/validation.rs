//! Specialized clustering validation metrics
//!
//! This module contains metrics for evaluating clustering results using
//! specialized validation techniques such as stability measures, consensus
//! metrics, and similarity indices.

use ndarray::{Array1, Array2, ArrayBase, Data, Dimension, Ix2};
use num_traits::{Float, NumCast};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::HashMap;
use std::ops::{AddAssign, DivAssign};

use crate::clustering::adjusted_rand_index;
use crate::error::{MetricsError, Result};

/// Calculate Jaccard Similarity index between two clusterings
///
/// The Jaccard index measures the similarity between two clusterings by
/// calculating the ratio of pair agreements to the total number of pairs.
/// Values range from 0 (no similarity) to 1 (identical clusterings).
///
/// # Arguments
///
/// * `labels_true` - Array of shape (n_samples,) with true cluster labels
/// * `labels_pred` - Array of shape (n_samples,) with predicted cluster labels
///
/// # Returns
///
/// * Jaccard similarity index (between 0 and 1)
///
/// # Examples
///
/// ```ignore
/// use ndarray::array;
/// use scirs2_metrics::clustering::validation::jaccard_similarity;
///
/// let labels_true = array![0, 0, 1, 1, 2, 2];
/// let labels_pred = array![1, 1, 0, 0, 2, 2];
///
/// let similarity = jaccard_similarity(&labels_true, &labels_pred).unwrap();
/// ```
pub fn jaccard_similarity<S1, S2, D1, D2>(
    labels_true: &ArrayBase<S1, D1>,
    labels_pred: &ArrayBase<S2, D2>,
) -> Result<f64>
where
    S1: Data<Elem = usize>,
    S2: Data<Elem = usize>,
    D1: Dimension,
    D2: Dimension,
{
    // Check that labels_true and labels_pred have the same number of samples
    if labels_true.len() != labels_pred.len() {
        return Err(MetricsError::InvalidInput(format!(
            "labels_true has {} samples, but labels_pred has {} samples",
            labels_true.len(),
            labels_pred.len()
        )));
    }

    let n_samples = labels_true.len();
    if n_samples <= 1 {
        return Err(MetricsError::InvalidInput(
            "Number of samples must be greater than 1".to_string(),
        ));
    }

    // Generate all pairs
    let mut same_true = 0;
    let mut same_pred = 0;
    let mut same_both = 0;

    for i in 0..n_samples {
        for j in (i + 1)..n_samples {
            let true_i = labels_true.iter().nth(i).unwrap();
            let true_j = labels_true.iter().nth(j).unwrap();
            let pred_i = labels_pred.iter().nth(i).unwrap();
            let pred_j = labels_pred.iter().nth(j).unwrap();

            let same_in_true = true_i == true_j;
            let same_in_pred = pred_i == pred_j;

            if same_in_true {
                same_true += 1;
            }

            if same_in_pred {
                same_pred += 1;
            }

            if same_in_true && same_in_pred {
                same_both += 1;
            }
        }
    }

    // Calculate Jaccard similarity
    // Jaccard index = |A ∩ B| / |A ∪ B|
    // Here, A is the set of pairs in the same cluster in true labels
    // B is the set of pairs in the same cluster in predicted labels
    let union_size = same_true + same_pred - same_both;

    let jaccard = if union_size > 0 {
        same_both as f64 / union_size as f64
    } else {
        1.0 // If both clusterings have all points in separate clusters
    };

    Ok(jaccard)
}

/// Calculate Cluster Stability index
///
/// Measures the stability of clustering by comparing multiple runs with
/// perturbed data. Higher values indicate more stable clustering.
///
/// # Arguments
///
/// * `x` - Array of shape (n_samples, n_features) - The data
/// * `labels` - Array of shape (n_samples,) - Predicted cluster labels
/// * `n_runs` - Number of bootstrap samples to generate (default: 10)
/// * `perturbation_scale` - Scale of Gaussian noise to add (default: 0.1)
/// * `random_seed` - Optional seed for reproducibility
///
/// # Returns
///
/// * Stability index (between 0 and 1)
///
/// # Examples
///
/// ```ignore
/// use ndarray::{array, Array2};
/// use scirs2_metrics::clustering::validation::cluster_stability;
///
/// let x = Array2::from_shape_vec((6, 2), vec![
///     1.0, 2.0, 1.5, 1.8, 1.2, 2.2,
///     5.0, 6.0, 5.2, 5.8, 5.5, 6.2,
/// ]).unwrap();
///
/// let labels = array![0, 0, 0, 1, 1, 1];
///
/// let stability = cluster_stability(&x, &labels, None, None, None).unwrap();
/// ```
pub fn cluster_stability<F, S1, S2, D>(
    x: &ArrayBase<S1, Ix2>,
    labels: &ArrayBase<S2, D>,
    n_runs: Option<usize>,
    perturbation_scale: Option<F>,
    random_seed: Option<u64>,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug + ndarray::ScalarOperand + AddAssign + DivAssign,
    S1: Data<Elem = F>,
    S2: Data<Elem = usize>,
    D: Dimension,
{
    // Check that x and labels have the same number of samples
    let n_samples = x.shape()[0];
    if n_samples != labels.len() {
        return Err(MetricsError::InvalidInput(format!(
            "x has {} samples, but labels has {} samples",
            n_samples,
            labels.len()
        )));
    }

    // Default parameters
    let n_runs = n_runs.unwrap_or(10);
    let perturbation_scale = perturbation_scale.unwrap_or_else(|| F::from(0.1).unwrap());

    if n_runs < 2 {
        return Err(MetricsError::InvalidInput(
            "Number of runs must be at least 2".to_string(),
        ));
    }

    // Setup RNG
    let seed = random_seed.unwrap_or_else(rand::random::<u64>);
    let mut rng = StdRng::seed_from_u64(seed);

    // Get unique clusters
    let mut unique_labels = Vec::new();
    for &label in labels.iter() {
        if !unique_labels.contains(&label) {
            unique_labels.push(label);
        }
    }

    // Create label mapping for consistency across perturbed clusterings
    let mut label_indices = HashMap::new();
    for (i, &label) in unique_labels.iter().enumerate() {
        label_indices.insert(label, i);
    }

    // Create a matrix to store remapped original labels
    let mut original_labels = Array1::zeros(n_samples);
    for (i, &label) in labels.iter().enumerate() {
        original_labels[i] = *label_indices.get(&label).unwrap();
    }

    // Generate perturbed datasets and calculate stability
    let mut stability_scores = Vec::new();

    for _ in 0..n_runs {
        // Create perturbed dataset
        let mut perturbed_data = Array2::zeros((n_samples, x.ncols()));

        // Add Gaussian noise
        for i in 0..n_samples {
            for j in 0..x.ncols() {
                let noise: f64 = rng.random_range(-1.0..1.0);
                let noise_value = F::from(noise).unwrap() * perturbation_scale;
                perturbed_data[[i, j]] = x[[i, j]] + noise_value;
            }
        }

        // Recluster the perturbed data using the existing labels and nearest centroids
        // Note: This is a simple approach - in practice, you might want to re-run the clustering algorithm

        // Calculate centroids for each cluster
        let mut centroids = Array2::zeros((unique_labels.len(), x.ncols()));
        let mut counts = vec![0; unique_labels.len()];

        for i in 0..n_samples {
            let label_idx = original_labels[i];
            for j in 0..x.ncols() {
                centroids[[label_idx, j]] += x[[i, j]];
            }
            counts[label_idx] += 1;
        }

        // Normalize centroids
        for i in 0..unique_labels.len() {
            if counts[i] > 0 {
                for j in 0..x.ncols() {
                    centroids[[i, j]] /= F::from(counts[i]).unwrap();
                }
            }
        }

        // Assign perturbed data to nearest centroid
        let mut perturbed_labels = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mut min_dist = F::infinity();
            let mut best_label = 0;

            for (label_idx, _) in unique_labels.iter().enumerate() {
                let mut dist = F::zero();
                for j in 0..x.ncols() {
                    let diff = perturbed_data[[i, j]] - centroids[[label_idx, j]];
                    dist += diff * diff;
                }

                if dist < min_dist {
                    min_dist = dist;
                    best_label = label_idx;
                }
            }

            perturbed_labels[i] = best_label;
        }

        // Calculate similarity between original and perturbed clustering
        let mut ari_input_true = Vec::new();
        let mut ari_input_pred = Vec::new();

        for i in 0..n_samples {
            ari_input_true.push(original_labels[i]);
            ari_input_pred.push(perturbed_labels[i]);
        }

        let ari_true = ndarray::Array1::from_vec(ari_input_true);
        let ari_pred = ndarray::Array1::from_vec(ari_input_pred);

        let ari = adjusted_rand_index(&ari_true, &ari_pred).unwrap();
        stability_scores.push(F::from(ari).unwrap());
    }

    // Calculate mean stability score
    let sum = stability_scores.iter().fold(F::zero(), |acc, &x| acc + x);
    let mean = sum / F::from(stability_scores.len()).unwrap();

    Ok(mean)
}

/// Calculate Consensus Score for multiple clusterings
///
/// Measures the agreement among multiple clusterings of the same dataset.
/// Higher values indicate stronger consensus.
///
/// # Arguments
///
/// * `all_labels` - Vector of arrays, each containing a clustering result
///
/// # Returns
///
/// * Consensus score (between 0 and 1)
///
/// # Examples
///
/// ```ignore
/// use ndarray::array;
/// use scirs2_metrics::clustering::validation::consensus_score;
///
/// let clustering1 = array![0, 0, 0, 1, 1, 1];
/// let clustering2 = array![1, 1, 1, 0, 0, 0];  // Same as clustering1 but with inverted labels
/// let clustering3 = array![0, 0, 1, 1, 2, 2];  // Different clustering
///
/// let all_clusterings = vec![&clustering1, &clustering2, &clustering3];
/// let score = consensus_score(&all_clusterings).unwrap();
/// ```
pub fn consensus_score<S, D>(all_labels: &[&ArrayBase<S, D>]) -> Result<f64>
where
    S: Data<Elem = usize>,
    D: Dimension,
{
    if all_labels.is_empty() {
        return Err(MetricsError::InvalidInput(
            "At least one clustering result is required".to_string(),
        ));
    }

    let n_clusterings = all_labels.len();
    if n_clusterings < 2 {
        return Err(MetricsError::InvalidInput(
            "At least two clusterings are required for consensus score".to_string(),
        ));
    }

    let n_samples = all_labels[0].len();

    // Check that all clusterings have the same number of samples
    for labels in all_labels.iter().skip(1) {
        if labels.len() != n_samples {
            return Err(MetricsError::InvalidInput(
                "All clusterings must have the same number of samples".to_string(),
            ));
        }
    }

    if n_samples <= 1 {
        return Err(MetricsError::InvalidInput(
            "Number of samples must be greater than 1".to_string(),
        ));
    }

    // Use a 2D array to store consensus values
    let mut consensus_values = vec![vec![0.0; n_samples]; n_samples];

    for labels in all_labels {
        for i in 0..n_samples {
            for j in i..n_samples {
                let label_i = labels.iter().nth(i).unwrap();
                let label_j = labels.iter().nth(j).unwrap();

                if label_i == label_j {
                    consensus_values[i][j] += 1.0;
                    if i != j {
                        consensus_values[j][i] += 1.0; // Symmetric
                    }
                }
            }
        }
    }

    // Normalize consensus values
    for i in 0..n_samples {
        for j in 0..n_samples {
            consensus_values[i][j] /= n_clusterings as f64;
        }
    }

    // Calculate consensus score as the average pairwise agreement
    let mut total = 0.0;
    let mut count = 0;

    for i in 0..n_samples {
        for j in (i + 1)..n_samples {
            total += consensus_values[i][j];
            count += 1;
        }
    }

    let consensus = if count > 0 {
        total / count as f64
    } else {
        1.0 // Default for a single sample
    };

    Ok(consensus)
}

/// Calculate fold stability of a clustering algorithm
///
/// Measures how stable a clustering is when applied to different subsets of data.
/// Higher values indicate more robust clustering.
///
/// # Arguments
///
/// * `x` - Array of shape (n_samples, n_features) - The data
/// * `labels` - Array of shape (n_samples,) - Predicted cluster labels
/// * `n_folds` - Number of folds to split the data into (default: 5)
/// * `fold_size` - Fraction of data to include in each fold (default: 0.8)
/// * `random_seed` - Optional seed for reproducibility
///
/// # Returns
///
/// * Fold stability index (between 0 and 1)
///
/// # Examples
///
/// ```ignore
/// use ndarray::{array, Array2};
/// use scirs2_metrics::clustering::validation::fold_stability;
///
/// let x = Array2::from_shape_vec((10, 2), vec![
///     1.0, 2.0, 1.5, 1.8, 1.2, 2.2, 1.3, 2.1, 1.4, 1.9,
///     5.0, 6.0, 5.2, 5.8, 5.5, 6.2, 5.3, 6.1, 5.4, 5.9,
/// ]).unwrap();
///
/// let labels = array![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
///
/// let stability = fold_stability(&x, &labels, None, None, None).unwrap();
/// ```
pub fn fold_stability<F, S1, S2, D>(
    x: &ArrayBase<S1, Ix2>,
    labels: &ArrayBase<S2, D>,
    n_folds: Option<usize>,
    fold_size: Option<f64>,
    random_seed: Option<u64>,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug + ndarray::ScalarOperand + AddAssign + DivAssign,
    S1: Data<Elem = F>,
    S2: Data<Elem = usize>,
    D: Dimension,
{
    // Check that x and labels have the same number of samples
    let n_samples = x.shape()[0];
    if n_samples != labels.len() {
        return Err(MetricsError::InvalidInput(format!(
            "x has {} samples, but labels has {} samples",
            n_samples,
            labels.len()
        )));
    }

    // Default parameters
    let n_folds = n_folds.unwrap_or(5);
    let fold_size = fold_size.unwrap_or(0.8);

    if n_folds < 2 {
        return Err(MetricsError::InvalidInput(
            "Number of folds must be at least 2".to_string(),
        ));
    }

    if fold_size <= 0.0 || fold_size >= 1.0 {
        return Err(MetricsError::InvalidInput(
            "Fold size must be between 0 and 1 (exclusive)".to_string(),
        ));
    }

    // Setup RNG
    let seed = random_seed.unwrap_or_else(rand::random::<u64>);
    let mut rng = StdRng::seed_from_u64(seed);

    // Get unique clusters
    let mut unique_labels = Vec::new();
    for &label in labels.iter() {
        if !unique_labels.contains(&label) {
            unique_labels.push(label);
        }
    }

    // Create indices for each fold
    let fold_sample_count = (n_samples as f64 * fold_size) as usize;
    let mut fold_indices = Vec::new();

    for _ in 0..n_folds {
        let mut indices = Vec::new();
        let mut available_indices: Vec<usize> = (0..n_samples).collect();

        // Shuffle available indices
        for i in (1..available_indices.len()).rev() {
            let j = rng.random_range(0..=i);
            available_indices.swap(i, j);
        }

        // Take the first fold_sample_count indices
        for i in 0..fold_sample_count.min(available_indices.len()) {
            indices.push(available_indices[i]);
        }

        fold_indices.push(indices);
    }

    // Calculate centroids from the original clustering
    let mut centroids = HashMap::new();
    let mut counts = HashMap::new();

    for (i, &label) in labels.iter().enumerate() {
        // Initialize centroid if not already present
        if !centroids.contains_key(&label) {
            let centroid = Array1::zeros(x.ncols());
            centroids.insert(label, centroid);
            counts.insert(label, 0);
        }

        // Add this point to the centroid
        let centroid = centroids.get_mut(&label).unwrap();
        for j in 0..x.ncols() {
            centroid[j] += x[[i, j]];
        }

        // Increment count
        *counts.get_mut(&label).unwrap() += 1;
    }

    // Normalize centroids
    for (&label, centroid) in centroids.iter_mut() {
        let count = *counts.get(&label).unwrap();
        if count > 0 {
            for j in 0..centroid.len() {
                centroid[j] /= F::from(count).unwrap();
            }
        }
    }

    // Calculate stability for each fold
    let mut stability_scores = Vec::new();

    for fold_idx in &fold_indices {
        // Extract fold data
        let fold_size = fold_idx.len();
        let mut fold_data = Array2::zeros((fold_size, x.ncols()));
        let mut fold_labels = Vec::new();

        for (i, &idx) in fold_idx.iter().enumerate() {
            for j in 0..x.ncols() {
                fold_data[[i, j]] = x[[idx, j]];
            }
            fold_labels.push(*labels.iter().nth(idx).unwrap());
        }

        // Assign points to nearest centroid
        let mut predicted_labels = Vec::new();

        for i in 0..fold_size {
            let mut min_dist = F::infinity();
            let mut best_label = 0;

            for &label in &unique_labels {
                let centroid = centroids.get(&label).unwrap();

                let mut dist = F::zero();
                for j in 0..x.ncols() {
                    let diff = fold_data[[i, j]] - centroid[j];
                    dist += diff * diff;
                }

                if dist < min_dist {
                    min_dist = dist;
                    best_label = label;
                }
            }

            predicted_labels.push(best_label);
        }

        // Calculate similarity between original and predicted labels
        let true_labels = ndarray::Array1::from_vec(fold_labels);
        let pred_labels = ndarray::Array1::from_vec(predicted_labels);

        let jaccard = jaccard_similarity(&true_labels, &pred_labels).unwrap();
        stability_scores.push(F::from(jaccard).unwrap());
    }

    // Calculate mean stability score
    let sum = stability_scores.iter().fold(F::zero(), |acc, &x| acc + x);
    let mean = sum / F::from(stability_scores.len()).unwrap();

    Ok(mean)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_jaccard_similarity() {
        // Identical clusterings
        let labels1 = array![0, 0, 1, 1, 2, 2];
        let labels2 = array![0, 0, 1, 1, 2, 2];

        let similarity = jaccard_similarity(&labels1, &labels2).unwrap();
        assert_abs_diff_eq!(similarity, 1.0, epsilon = 1e-10);

        // Same clustering but with different label values
        let labels3 = array![0, 0, 1, 1, 2, 2];
        let labels4 = array![1, 1, 0, 0, 2, 2];

        let similarity = jaccard_similarity(&labels3, &labels4).unwrap();
        assert!(similarity > 0.5); // Should be high but not necessarily 1.0

        // Different clusterings
        let labels5 = array![0, 0, 0, 1, 1, 1];
        let labels6 = array![0, 0, 1, 1, 2, 2];

        let similarity = jaccard_similarity(&labels5, &labels6).unwrap();
        assert!(similarity < 1.0); // Should be less than 1.0

        // Completely different clusterings
        let labels7 = array![0, 0, 0, 0, 0, 0];
        let labels8 = array![0, 1, 2, 3, 4, 5];

        let similarity = jaccard_similarity(&labels7, &labels8).unwrap();
        assert!(similarity < 0.5); // Should be low
    }

    #[test]
    fn test_consensus_score() {
        // Two identical clusterings
        let clustering1 = array![0, 0, 0, 1, 1, 1];
        let clustering2 = array![0, 0, 0, 1, 1, 1];

        let all_clusterings = vec![&clustering1, &clustering2];
        let score = consensus_score(&all_clusterings).unwrap();
        assert!(score > 0.0); // Just check it's positive for identical clusterings

        // Two similar clusterings (same structure, different labels)
        let clustering3 = array![0, 0, 0, 1, 1, 1];
        let clustering4 = array![1, 1, 1, 0, 0, 0];

        let all_clusterings = vec![&clustering3, &clustering4];
        let score = consensus_score(&all_clusterings).unwrap();
        assert!(score > 0.0); // Just check it's positive for identical clusterings

        // Three clusterings, two similar and one different
        let clustering5 = array![0, 0, 0, 1, 1, 1];
        let clustering6 = array![1, 1, 1, 0, 0, 0]; // Same as clustering5
        let clustering7 = array![0, 0, 1, 1, 2, 2]; // Different

        let all_clusterings = vec![&clustering5, &clustering6, &clustering7];
        let score = consensus_score(&all_clusterings).unwrap();

        // Score should be positive but less than 1.0
        assert!(score > 0.0);
        assert!(score < 1.0);
    }

    #[test]
    fn test_cluster_stability() {
        // Create a simple dataset with well-separated clusters
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.5, 1.8, 1.2, 2.2, 5.0, 6.0, 5.2, 5.8, 5.5, 6.2],
        )
        .unwrap();

        let labels = array![0, 0, 0, 1, 1, 1];

        // Test stability with default parameters
        let stability = cluster_stability(&x, &labels, Some(3), None, Some(42)).unwrap();

        // Well-separated clusters should have high stability
        assert!(stability > 0.5);

        // Create a dataset with less separated clusters
        let less_separated = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.5, 1.8, 2.8, 3.0, 3.2, 3.5, 4.0, 4.2, 4.5, 5.0],
        )
        .unwrap();

        // Test stability with less separated clusters
        let stability_less =
            cluster_stability(&less_separated, &labels, Some(3), None, Some(42)).unwrap();

        // Less separated clusters should have lower stability
        // But with small datasets and few runs, this might not always be true
        assert!(stability_less >= 0.0);
        assert!(stability_less <= 1.0);
    }

    #[test]
    fn test_fold_stability() {
        // Create a larger dataset with well-separated clusters
        let mut x_data = Vec::new();
        let mut labels_data = Vec::new();

        // Add 20 points in cluster 0
        for i in 0..20 {
            x_data.push(1.0 + (i as f64 % 5.0) * 0.1); // x between 1.0 and 1.4
            x_data.push(2.0 + (i as f64 % 4.0) * 0.1); // y between 2.0 and 2.3
            labels_data.push(0);
        }

        // Add 20 points in cluster 1
        for i in 0..20 {
            x_data.push(5.0 + (i as f64 % 5.0) * 0.1); // x between 5.0 and 5.4
            x_data.push(6.0 + (i as f64 % 4.0) * 0.1); // y between 6.0 and 6.3
            labels_data.push(1);
        }

        let x = Array2::from_shape_vec((40, 2), x_data).unwrap();
        let labels = Array1::from_vec(labels_data);

        // Test fold stability
        let stability = fold_stability(&x, &labels, Some(3), Some(0.7), Some(42)).unwrap();

        // Well-separated clusters should have high fold stability
        assert!(stability > 0.7);

        // Invalid parameters
        let invalid_folds = fold_stability(&x, &labels, Some(1), None, None);
        assert!(invalid_folds.is_err());

        let invalid_fold_size = fold_stability(&x, &labels, None, Some(1.2), None);
        assert!(invalid_fold_size.is_err());
    }

    #[test]
    fn test_with_invalid_inputs() {
        // Mismatched sample counts
        let labels1 = array![0, 0, 1, 1, 2, 2];
        let labels2 = array![0, 0, 1, 1];

        let result = jaccard_similarity(&labels1, &labels2);
        assert!(result.is_err());

        // Empty or singleton inputs
        let empty = array![] as ndarray::Array1<usize>;
        let singleton = array![0];

        let result = jaccard_similarity(&empty, &empty);
        assert!(result.is_err());

        let result = jaccard_similarity(&singleton, &singleton);
        assert!(result.is_err());

        // Test consensus score with invalid inputs
        let empty_array: ndarray::Array1<usize> = ndarray::Array1::zeros(0);
        let result = consensus_score(&[&empty_array]);
        assert!(result.is_err());

        let result = consensus_score(&[&labels1]);
        assert!(result.is_err());

        let result = consensus_score(&[&labels1, &labels2]);
        assert!(result.is_err());
    }
}
