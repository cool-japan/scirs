//! Mini-Batch K-means clustering implementation
//!
//! This module provides an implementation of the Mini-Batch K-means algorithm,
//! a variant of k-means that uses mini-batches to reduce computation time while
//! still attempting to optimize the same objective function.
//!
//! Mini-Batch K-means is much faster than standard K-means for large datasets
//! and provides results that are generally close to those of the standard algorithm.

use ndarray::{s, Array1, Array2, ArrayView2};
use num_traits::{Float, FromPrimitive};
use rand::{Rng, SeedableRng};
use std::fmt::Debug;

use super::{euclidean_distance, kmeans_plus_plus};
use crate::error::{ClusteringError, Result};

/// Options for Mini-Batch K-means clustering
#[derive(Debug, Clone)]
pub struct MiniBatchKMeansOptions<F: Float> {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Size of mini-batches
    pub batch_size: usize,
    /// Convergence threshold for centroid movement
    pub tol: F,
    /// Random seed for initialization and batch sampling
    pub random_seed: Option<u64>,
    /// Number of iterations without improvement before stopping
    pub max_no_improvement: usize,
    /// Number of samples to use for initialization
    pub init_size: Option<usize>,
    /// Ratio of samples that should be reassigned to prevent empty clusters
    pub reassignment_ratio: F,
}

impl<F: Float + FromPrimitive> Default for MiniBatchKMeansOptions<F> {
    fn default() -> Self {
        Self {
            max_iter: 100,
            batch_size: 1024,
            tol: F::from(1e-4).unwrap(),
            random_seed: None,
            max_no_improvement: 10,
            init_size: None,
            reassignment_ratio: F::from(0.01).unwrap(),
        }
    }
}

/// Mini-Batch K-means clustering algorithm
///
/// # Arguments
///
/// * `data` - Input data (n_samples × n_features)
/// * `k` - Number of clusters
/// * `options` - Optional parameters
///
/// # Returns
///
/// * Tuple of (centroids, labels) where:
///   - centroids: Array of shape (k × n_features)
///   - labels: Array of shape (n_samples,) with cluster assignments
///
/// # Examples
///
/// ```
/// use ndarray::{Array2, ArrayView2};
/// use scirs2_cluster::vq::minibatch_kmeans;
///
/// let data = Array2::from_shape_vec((6, 2), vec![
///     1.0, 2.0,
///     1.2, 1.8,
///     0.8, 1.9,
///     3.7, 4.2,
///     3.9, 3.9,
///     4.2, 4.1,
/// ]).unwrap();
///
/// let (centroids, labels) = minibatch_kmeans(ArrayView2::from(&data), 2, None).unwrap();
/// ```
#[allow(dead_code)]
pub fn minibatch_kmeans<F>(
    data: ArrayView2<F>,
    k: usize,
    options: Option<MiniBatchKMeansOptions<F>>,
) -> Result<(Array2<F>, Array1<usize>)>
where
    F: Float + FromPrimitive + Debug + std::iter::Sum,
{
    // Input validation
    if k == 0 {
        return Err(ClusteringError::InvalidInput(
            "Number of clusters must be greater than 0".to_string(),
        ));
    }

    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];

    if n_samples == 0 {
        return Err(ClusteringError::InvalidInput(
            "Input data is empty".to_string(),
        ));
    }

    if k > n_samples {
        return Err(ClusteringError::InvalidInput(format!(
            "Number of clusters ({}) cannot be greater than number of data points ({})",
            k, n_samples
        )));
    }

    let opts = options.unwrap_or_default();

    // Setup RNG
    let mut rng = match opts.random_seed {
        Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
        None => rand::rngs::StdRng::seed_from_u64(rand::rng().random()),
    };

    // Determine initialization size
    let init_size = opts.init_size.unwrap_or_else(|| {
        let default_size = 3 * opts.batch_size;
        if default_size < 3 * k {
            default_size
        } else {
            3 * k
        }
    });

    let init_size = init_size.min(n_samples);

    // Initialize centroids using kmeans++
    let centroids = if init_size < n_samples {
        // Sample init_size data points for initialization (simpler method for this example)
        let mut indices = Vec::with_capacity(init_size);
        for _ in 0..init_size {
            indices.push(rng.random_range(0..n_samples));
        }

        let init_data =
            Array2::from_shape_fn((init_size, n_features), |(i, j)| data[[indices[i], j]]);
        kmeans_plus_plus(init_data.view(), k, opts.random_seed)?
    } else {
        // Use all data points for initialization
        kmeans_plus_plus(data, k, opts.random_seed)?
    };

    // Initialize variables for optimization
    let mut centroids = centroids;
    let mut counts = Array1::ones(k); // Initialize counts to avoid division by zero

    // Variables for convergence detection
    let mut ewa_inertia = None; // Exponentially weighted average of inertia
    let mut no_improvement_count = 0;
    let mut best_inertia = F::infinity();
    let mut prev_centers: Option<Array2<F>> = None;

    // Mini-batch optimization
    for iter in 0..opts.max_iter {
        // Sample a mini-batch
        let batch_size = opts.batch_size.min(n_samples);
        let mut batch_indices = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            batch_indices.push(rng.random_range(0..n_samples));
        }

        // Perform mini-batch step
        let (batch_inertia, has_converged) =
            mini_batch_step(&data, &batch_indices, &mut centroids, &mut counts, &opts)?;

        // If this is the last iteration, assign all points to clusters for final labeling
        // We don't need to do this on every iteration, just for the final result
        if iter == opts.max_iter - 1 {
            // This will be used only for the final return value
            let (_new_labels_) = assign_labels(data, centroids.view())?;
            // We don't store this since we'll recompute it at the end anyway
        }

        // Update exponentially weighted average of inertia
        let ewa_factor = F::from(0.7).unwrap(); // Smoothing factor for EWA
        let current_ewa = match ewa_inertia {
            Some(prev_ewa) => prev_ewa * ewa_factor + batch_inertia * (F::one() - ewa_factor),
            None => batch_inertia,
        };
        ewa_inertia = Some(current_ewa);

        // Check for convergence based on inertia
        if current_ewa < best_inertia {
            best_inertia = current_ewa;
            no_improvement_count = 0;
        } else {
            no_improvement_count += 1;
        }

        // Check for convergence based on centroid movement
        if let Some(prev) = prev_centers {
            let mut center_shift = F::zero();
            for i in 0..k {
                let dist = euclidean_distance(centroids.slice(s![i, ..]), prev.slice(s![i, ..]));
                center_shift = center_shift + dist;
            }

            // Normalize by number of centroids and features
            center_shift = center_shift / F::from(k).unwrap();

            if center_shift < opts.tol {
                // Converged based on centroid movement
                break;
            }
        }

        // Store current centroids for next iteration
        prev_centers = Some(centroids.clone());

        // Check for early stopping
        if no_improvement_count >= opts.max_no_improvement {
            break;
        }

        // If convergence detected in mini-batch step
        if has_converged {
            break;
        }
    }

    // Final label assignment
    let (final_labels, _) = assign_labels(data, centroids.view())?;

    Ok((centroids, final_labels))
}

/// Performs a single Mini-Batch K-means step
///
/// # Arguments
///
/// * `data` - Input data
/// * `batch_indices` - Indices of samples in the current mini-batch
/// * `centroids` - Current centroids (modified in-place)
/// * `counts` - Counts of samples assigned to each centroid (modified in-place)
/// * `opts` - Algorithm options
///
/// # Returns
///
/// * Tuple of (batch_inertia, has_converged)
#[allow(dead_code)]
fn mini_batch_step<F>(
    data: &ArrayView2<F>,
    batch_indices: &[usize],
    centroids: &mut Array2<F>,
    counts: &mut Array1<F>,
    opts: &MiniBatchKMeansOptions<F>,
) -> Result<(F, bool)>
where
    F: Float + FromPrimitive + Debug,
{
    let k = centroids.shape()[0];
    let n_features = centroids.shape()[1];
    let batch_size = batch_indices.len();

    // Initialize mini-batch specific variables
    let mut closest_distances = Array1::from_elem(batch_size, F::infinity());
    let mut closest_centers = Array1::zeros(batch_size);
    let mut inertia = F::zero();

    // Assign samples to closest centroids
    for (i, &sample_idx) in batch_indices.iter().enumerate() {
        let sample = data.slice(s![sample_idx, ..]);

        // Find closest centroid
        let mut min_dist = F::infinity();
        let mut min_idx = 0;

        for j in 0..k {
            let dist = euclidean_distance(sample, centroids.slice(s![j, ..]));
            if dist < min_dist {
                min_dist = dist;
                min_idx = j;
            }
        }

        closest_centers[i] = min_idx;
        closest_distances[i] = min_dist;
        inertia = inertia + min_dist * min_dist;
    }

    // Update centroids based on mini-batch assignments
    for i in 0..batch_size {
        let center_idx = closest_centers[i];
        let sample_idx = batch_indices[i];
        let sample = data.slice(s![sample_idx, ..]);

        // Incremental update of centroid
        let count = counts[center_idx];
        let learning_rate = F::one() / (count + F::one()); // Decrease learning rate as count increases

        for j in 0..n_features {
            centroids[[center_idx, j]] =
                centroids[[center_idx, j]] * (F::one() - learning_rate) + sample[j] * learning_rate;
        }

        counts[center_idx] = count + F::one();
    }

    // Handle reassignment of small or empty clusters
    let mut has_empty = false;
    let max_count = counts.fold(F::zero(), |a, &b| a.max(b));
    let reassign_threshold = max_count * opts.reassignment_ratio;

    for i in 0..k {
        if counts[i] < reassign_threshold {
            has_empty = true;

            // Find the point furthest from its centroid in this batch
            let mut max_dist = F::zero();
            let mut max_idx = 0;

            for j in 0..batch_size {
                if closest_distances[j] > max_dist {
                    max_dist = closest_distances[j];
                    max_idx = j;
                }
            }

            // Reassign this centroid to the furthest point
            if max_dist > F::zero() {
                let sample_idx = batch_indices[max_idx];
                let sample = data.slice(s![sample_idx, ..]);

                for j in 0..n_features {
                    centroids[[i, j]] = sample[j];
                }

                // Reset count to a small value to prevent immediate reassignment
                counts[i] = counts[i].max(F::from(1.0).unwrap());

                // Update closest center and distance for this point
                closest_centers[max_idx] = i;
                closest_distances[max_idx] = F::zero();
            }
        }
    }

    // Normalize inertia by batch size
    inertia = inertia / F::from(batch_size).unwrap();

    // Check if we have converged
    let has_converged = !has_empty && inertia < opts.tol;

    Ok((inertia, has_converged))
}

/// Assigns each sample in the dataset to its closest centroid
///
/// # Arguments
///
/// * `data` - Input data
/// * `centroids` - Current centroids
///
/// # Returns
///
/// * Tuple of (labels, distances)
#[allow(dead_code)]
fn assign_labels<F>(
    data: ArrayView2<F>,
    centroids: ArrayView2<F>,
) -> Result<(Array1<usize>, Array1<F>)>
where
    F: Float + FromPrimitive + Debug,
{
    let n_samples = data.shape()[0];
    let n_clusters = centroids.shape()[0];

    let mut labels = Array1::zeros(n_samples);
    let mut distances = Array1::zeros(n_samples);

    for i in 0..n_samples {
        let sample = data.slice(s![i, ..]);
        let mut min_dist = F::infinity();
        let mut min_idx = 0;

        for j in 0..n_clusters {
            let centroid = centroids.slice(s![j, ..]);
            let dist = euclidean_distance(sample, centroid);

            if dist < min_dist {
                min_dist = dist;
                min_idx = j;
            }
        }

        labels[i] = min_idx;
        distances[i] = min_dist;
    }

    Ok((labels, distances))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_minibatch_kmeans_simple() {
        // Create a simple dataset with clear clusters
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.2, 1.8, 0.8, 1.9, 4.0, 5.0, 4.2, 4.8, 3.9, 5.1],
        )
        .unwrap();

        // Run mini-batch k-means with k=2
        let options = MiniBatchKMeansOptions {
            max_iter: 10,
            batch_size: 3,
            random_seed: Some(42), // For reproducibility
            ..Default::default()
        };

        let (centroids, labels) = minibatch_kmeans(data.view(), 2, Some(options)).unwrap();

        // Check dimensions
        assert_eq!(centroids.shape(), &[2, 2]);
        assert_eq!(labels.shape(), &[6]);

        // Check that we have 2 unique labels
        let unique_labels: Vec<_> = labels
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        assert_eq!(unique_labels.len(), 2);

        // Check that the first 3 points are in one cluster and the last 3 in another
        let first_label = labels[0];
        assert_eq!(labels[1], first_label);
        assert_eq!(labels[2], first_label);

        let second_label = labels[3];
        assert_eq!(labels[4], second_label);
        assert_eq!(labels[5], second_label);

        // First cluster should be around (1, 2)
        let cluster1_idx = if first_label == 0 { 0 } else { 1 };
        assert!((centroids[[cluster1_idx, 0]] - 1.0).abs() < 0.5);
        assert!((centroids[[cluster1_idx, 1]] - 2.0).abs() < 0.5);

        // Second cluster should be around (4, 5)
        let cluster2_idx = if first_label == 0 { 1 } else { 0 };
        assert!((centroids[[cluster2_idx, 0]] - 4.0).abs() < 0.5);
        assert!((centroids[[cluster2_idx, 1]] - 5.0).abs() < 0.5);
    }

    #[test]
    fn test_minibatch_kmeans_empty_clusters() {
        // Create a dataset where empty clusters could occur
        let data = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.1, 1.1, 1.2, 1.0, 1.0, 1.2, 5.0, 5.0, 5.1, 5.1, 5.2, 5.0, 5.0, 5.2,
            ],
        )
        .unwrap();

        // Run mini-batch k-means with k=3 (which would likely lead to an empty cluster)
        let options = MiniBatchKMeansOptions {
            max_iter: 20,
            batch_size: 4,
            random_seed: Some(42),   // For reproducibility
            reassignment_ratio: 0.1, // Higher reassignment to test this feature
            ..Default::default()
        };

        let (centroids, labels) = minibatch_kmeans(data.view(), 3, Some(options)).unwrap();

        // Check dimensions
        assert_eq!(centroids.shape(), &[3, 2]);
        assert_eq!(labels.shape(), &[8]);

        // We should have at most 3 clusters
        let unique_labels: Vec<_> = labels
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        assert!(unique_labels.len() <= 3);

        // Every centroid should have at least one point assigned to it
        let mut centroid_counts = [0; 3];
        for &label in labels.iter() {
            centroid_counts[label] += 1;
        }

        // We might not have all 3 clusters used due to reassignment
        // but there should be no empty clusters in the output
        for &count in centroid_counts.iter() {
            assert!(count > 0);
        }
    }
}
