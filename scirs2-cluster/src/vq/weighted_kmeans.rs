//! Weighted K-means clustering implementation
//!
//! This module provides K-means clustering with support for weighted samples,
//! where each data point can have a different importance weight.

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, FromPrimitive};
use rand::Rng;
use std::fmt::Debug;

use super::{euclidean_distance, kmeans_init, KMeansInit};
use crate::error::{ClusteringError, Result};

/// Options for weighted K-means clustering
#[derive(Debug, Clone)]
pub struct WeightedKMeansOptions<F: Float> {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence threshold for centroid movement
    pub tol: F,
    /// Random seed for initialization
    pub random_seed: Option<u64>,
    /// Number of different initializations to try
    pub n_init: usize,
    /// Method to use for centroid initialization
    pub init_method: KMeansInit,
}

impl<F: Float + FromPrimitive> Default for WeightedKMeansOptions<F> {
    fn default() -> Self {
        Self {
            max_iter: 300,
            tol: F::from(1e-4).unwrap(),
            random_seed: None,
            n_init: 10,
            init_method: KMeansInit::KMeansPlusPlus,
        }
    }
}

/// Weighted K-means clustering algorithm
///
/// This algorithm allows each data point to have a different weight,
/// which affects both the centroid calculation and the overall objective function.
///
/// # Arguments
///
/// * `data` - Input data (n_samples × n_features)
/// * `weights` - Sample weights (n_samples,). Higher weights mean more important samples
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
/// use ndarray::{Array1, Array2};
/// use scirs2_cluster::vq::weighted_kmeans;
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
/// // Give higher weight to the first three points
/// let weights = Array1::from_vec(vec![2.0, 2.0, 2.0, 1.0, 1.0, 1.0]);
///
/// let (centroids, labels) = weighted_kmeans(data.view(), weights.view(), 2, None).unwrap();
/// ```
pub fn weighted_kmeans<F>(
    data: ArrayView2<F>,
    weights: ArrayView1<F>,
    k: usize,
    options: Option<WeightedKMeansOptions<F>>,
) -> Result<(Array2<F>, Array1<usize>)>
where
    F: Float + FromPrimitive + Debug + std::iter::Sum,
{
    if k == 0 {
        return Err(ClusteringError::InvalidInput(
            "Number of clusters must be greater than 0".to_string(),
        ));
    }

    let n_samples = data.shape()[0];
    if n_samples == 0 {
        return Err(ClusteringError::InvalidInput(
            "Input data is empty".to_string(),
        ));
    }

    if weights.len() != n_samples {
        return Err(ClusteringError::InvalidInput(
            "Weights array must have the same length as the number of samples".to_string(),
        ));
    }

    if k > n_samples {
        return Err(ClusteringError::InvalidInput(format!(
            "Number of clusters ({}) cannot be greater than number of data points ({})",
            k, n_samples
        )));
    }

    // Check that all weights are non-negative
    for &weight in weights.iter() {
        if weight < F::zero() {
            return Err(ClusteringError::InvalidInput(
                "All weights must be non-negative".to_string(),
            ));
        }
    }

    let opts = options.unwrap_or_default();

    let mut best_centroids = None;
    let mut best_labels = None;
    let mut best_inertia = F::infinity();

    for _ in 0..opts.n_init {
        // Initialize centroids using the specified method
        let centroids = kmeans_init(data, k, Some(opts.init_method), opts.random_seed)?;

        // Run weighted k-means
        let (centroids, labels, inertia) =
            weighted_kmeans_single(data, weights, centroids.view(), &opts)?;

        if inertia < best_inertia {
            best_centroids = Some(centroids);
            best_labels = Some(labels);
            best_inertia = inertia;
        }
    }

    Ok((best_centroids.unwrap(), best_labels.unwrap()))
}

/// Run a single weighted k-means clustering iteration
fn weighted_kmeans_single<F>(
    data: ArrayView2<F>,
    weights: ArrayView1<F>,
    init_centroids: ArrayView2<F>,
    opts: &WeightedKMeansOptions<F>,
) -> Result<(Array2<F>, Array1<usize>, F)>
where
    F: Float + FromPrimitive + Debug + std::iter::Sum,
{
    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];
    let k = init_centroids.shape()[0];

    let mut centroids = init_centroids.to_owned();
    let mut labels = Array1::zeros(n_samples);
    let mut prev_centroid_diff = F::infinity();

    for _iter in 0..opts.max_iter {
        // Assign samples to nearest centroid
        let (new_labels, distances) = weighted_assign_labels(data, centroids.view())?;
        labels = new_labels;

        // Compute new centroids using weights
        let mut new_centroids = Array2::zeros((k, n_features));
        let mut total_weights = Array1::zeros(k);

        for i in 0..n_samples {
            let cluster = labels[i];
            let point = data.slice(s![i, ..]);
            let weight = weights[i];

            for j in 0..n_features {
                new_centroids[[cluster, j]] = new_centroids[[cluster, j]] + point[j] * weight;
            }

            total_weights[cluster] = total_weights[cluster] + weight;
        }

        // If a cluster is empty or has very low total weight, reinitialize it
        for i in 0..k {
            if total_weights[i] <= F::epsilon() {
                // Find the point with highest weight * distance to its centroid
                let mut max_score = F::zero();
                let mut far_idx = 0;

                for j in 0..n_samples {
                    let score = weights[j] * distances[j];
                    if score > max_score {
                        max_score = score;
                        far_idx = j;
                    }
                }

                // Move this point to the empty cluster
                for j in 0..n_features {
                    new_centroids[[i, j]] = data[[far_idx, j]];
                }

                total_weights[i] = weights[far_idx];
            } else {
                // Normalize by the total weight in the cluster
                for j in 0..n_features {
                    new_centroids[[i, j]] = new_centroids[[i, j]] / total_weights[i];
                }
            }
        }

        // Check for convergence
        let mut centroid_diff = F::zero();
        for i in 0..k {
            let dist =
                euclidean_distance(centroids.slice(s![i, ..]), new_centroids.slice(s![i, ..]));
            centroid_diff = centroid_diff + dist;
        }

        centroids = new_centroids;

        if centroid_diff <= opts.tol || centroid_diff >= prev_centroid_diff {
            break;
        }

        prev_centroid_diff = centroid_diff;
    }

    // Calculate weighted inertia (sum of weighted squared distances to nearest centroid)
    let mut inertia = F::zero();
    for i in 0..n_samples {
        let cluster = labels[i];
        let dist = euclidean_distance(data.slice(s![i, ..]), centroids.slice(s![cluster, ..]));
        inertia = inertia + weights[i] * dist * dist;
    }

    Ok((centroids, labels, inertia))
}

/// Assign samples to nearest centroids (same as regular assignment)
fn weighted_assign_labels<F>(
    data: ArrayView2<F>,
    centroids: ArrayView2<F>,
) -> Result<(Array1<usize>, Array1<F>)>
where
    F: Float + FromPrimitive + Debug,
{
    let n_samples = data.shape()[0];
    let k = centroids.shape()[0];

    let mut labels = Array1::zeros(n_samples);
    let mut distances = Array1::zeros(n_samples);

    for i in 0..n_samples {
        let point = data.slice(s![i, ..]);
        let mut min_dist = F::infinity();
        let mut closest_centroid = 0;

        for j in 0..k {
            let centroid = centroids.slice(s![j, ..]);
            let dist = euclidean_distance(point, centroid);

            if dist < min_dist {
                min_dist = dist;
                closest_centroid = j;
            }
        }

        labels[i] = closest_centroid;
        distances[i] = min_dist;
    }

    Ok((labels, distances))
}

/// Weighted K-means++ initialization
///
/// This uses the weighted version of k-means++ where the probability of selecting
/// a point as a centroid is proportional to its weight times its squared distance
/// to the nearest existing centroid.
///
/// # Arguments
///
/// * `data` - Input data (n_samples × n_features)
/// * `weights` - Sample weights (n_samples,)
/// * `k` - Number of clusters
/// * `random_seed` - Optional random seed
///
/// # Returns
///
/// * Array of shape (k × n_features) with initial centroids
pub fn weighted_kmeans_plus_plus<F>(
    data: ArrayView2<F>,
    weights: ArrayView1<F>,
    k: usize,
    _random_seed: Option<u64>,
) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + std::iter::Sum,
{
    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];

    if k == 0 || k > n_samples {
        return Err(ClusteringError::InvalidInput(format!(
            "Number of clusters ({}) must be between 1 and number of samples ({})",
            k, n_samples
        )));
    }

    if weights.len() != n_samples {
        return Err(ClusteringError::InvalidInput(
            "Weights array must have the same length as the number of samples".to_string(),
        ));
    }

    let mut rng = rand::rng();

    let mut centroids = Array2::zeros((k, n_features));

    // Choose the first centroid randomly with probability proportional to weights
    let total_weight: F = weights.iter().copied().sum();
    let mut cumulative_weights = Array1::zeros(n_samples);
    cumulative_weights[0] = weights[0] / total_weight;
    for i in 1..n_samples {
        cumulative_weights[i] = cumulative_weights[i - 1] + weights[i] / total_weight;
    }

    let rand_val = F::from(rng.random::<f64>()).unwrap();
    let mut first_idx = 0;
    for i in 0..n_samples {
        if rand_val <= cumulative_weights[i] {
            first_idx = i;
            break;
        }
    }

    for j in 0..n_features {
        centroids[[0, j]] = data[[first_idx, j]];
    }

    if k == 1 {
        return Ok(centroids);
    }

    // Choose remaining centroids using weighted k-means++ algorithm
    for i in 1..k {
        // Compute weighted squared distances to closest centroid for each point
        let mut weighted_distances = Array1::from_elem(n_samples, F::zero());

        for sample_idx in 0..n_samples {
            let sample = data.slice(s![sample_idx, ..]);
            let mut min_dist_sq = F::infinity();

            for centroid_idx in 0..i {
                let centroid = centroids.slice(s![centroid_idx, ..]);
                let dist = euclidean_distance(sample, centroid);
                let dist_sq = dist * dist;

                if dist_sq < min_dist_sq {
                    min_dist_sq = dist_sq;
                }
            }

            weighted_distances[sample_idx] = weights[sample_idx] * min_dist_sq;
        }

        // Normalize the weighted distances to create a probability distribution
        let sum_weighted_distances: F = weighted_distances.iter().copied().sum();
        if sum_weighted_distances <= F::epsilon() {
            // If all weighted distances are zero, use uniform distribution among remaining points
            let remaining_weight: F = weights.iter().copied().sum();
            for sample_idx in 0..n_samples {
                weighted_distances[sample_idx] = weights[sample_idx] / remaining_weight;
            }
        } else {
            weighted_distances.mapv_inplace(|d| d / sum_weighted_distances);
        }

        // Convert to cumulative distribution
        let mut cum_weighted_distances = weighted_distances.clone();
        for j in 1..n_samples {
            cum_weighted_distances[j] = cum_weighted_distances[j] + cum_weighted_distances[j - 1];
        }

        // Sample the next centroid based on the weighted probability distribution
        let rand_val = F::from(rng.random::<f64>()).unwrap();
        let mut next_idx = 0;

        for j in 0..n_samples {
            if rand_val <= cum_weighted_distances[j] {
                next_idx = j;
                break;
            }
        }

        // Add the new centroid
        for j in 0..n_features {
            centroids[[i, j]] = data[[next_idx, j]];
        }
    }

    Ok(centroids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, Array2};

    #[test]
    fn test_weighted_kmeans_simple() {
        // Create a simple dataset with clear clusters
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.2, 1.8, 0.8, 1.9, 4.0, 5.0, 4.2, 4.8, 3.9, 5.1],
        )
        .unwrap();

        // Equal weights (should behave like regular k-means)
        let weights = Array1::from_elem(6, 1.0);

        let options = WeightedKMeansOptions {
            n_init: 1,
            random_seed: Some(42),
            ..Default::default()
        };

        let (centroids, labels) =
            weighted_kmeans(data.view(), weights.view(), 2, Some(options)).unwrap();

        // Check dimensions
        assert_eq!(centroids.shape(), &[2, 2]);
        assert_eq!(labels.len(), 6);

        // Check that we have 2 clusters
        let unique_labels: Vec<_> = labels
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        assert_eq!(unique_labels.len(), 2);
    }

    #[test]
    fn test_weighted_kmeans_different_weights() {
        // Create a simple dataset
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.2, 1.8, 0.8, 1.9, 4.0, 5.0, 4.2, 4.8, 3.9, 5.1],
        )
        .unwrap();

        // Give higher weight to the first three points
        let weights = Array1::from_vec(vec![10.0, 10.0, 10.0, 1.0, 1.0, 1.0]);

        let options = WeightedKMeansOptions {
            n_init: 1,
            random_seed: Some(42),
            ..Default::default()
        };

        let (centroids, labels) =
            weighted_kmeans(data.view(), weights.view(), 2, Some(options)).unwrap();

        // Check dimensions
        assert_eq!(centroids.shape(), &[2, 2]);
        assert_eq!(labels.len(), 6);

        // The centroid of the first cluster should be closer to the weighted center of the first 3 points
        let first_cluster_label = labels[0];
        let first_centroid = if first_cluster_label == 0 { 0 } else { 1 };

        // The first cluster centroid should be close to the mean of the first 3 points
        // because they have much higher weights
        let expected_centroid_x = (1.0 * 10.0 + 1.2 * 10.0 + 0.8 * 10.0) / (10.0 + 10.0 + 10.0);
        let expected_centroid_y = (2.0 * 10.0 + 1.8 * 10.0 + 1.9 * 10.0) / (10.0 + 10.0 + 10.0);

        let actual_centroid_x = centroids[[first_centroid, 0]];
        let actual_centroid_y = centroids[[first_centroid, 1]];

        // The centroids should be close to the expected weighted means
        assert_abs_diff_eq!(actual_centroid_x, expected_centroid_x, epsilon = 0.2);
        assert_abs_diff_eq!(actual_centroid_y, expected_centroid_y, epsilon = 0.2);
    }

    #[test]
    fn test_weighted_kmeans_plus_plus() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.2, 1.8, 0.8, 1.9, 4.0, 5.0, 4.2, 4.8, 3.9, 5.1],
        )
        .unwrap();

        let weights = Array1::from_vec(vec![1.0, 1.0, 1.0, 10.0, 10.0, 10.0]);

        let centroids =
            weighted_kmeans_plus_plus(data.view(), weights.view(), 2, Some(42)).unwrap();

        // Check dimensions
        assert_eq!(centroids.shape(), &[2, 2]);

        // All centroid values should be finite
        for val in centroids.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_weighted_kmeans_zero_weights() {
        let data =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 1.2, 1.8, 4.0, 5.0, 4.2, 4.8]).unwrap();

        // Some zero weights should still work
        let weights = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0]);

        let options = WeightedKMeansOptions {
            n_init: 1,
            random_seed: Some(42),
            ..Default::default()
        };

        let result = weighted_kmeans(data.view(), weights.view(), 2, Some(options));
        assert!(result.is_ok());

        let (centroids, labels) = result.unwrap();
        assert_eq!(centroids.shape(), &[2, 2]);
        assert_eq!(labels.len(), 4);
    }

    #[test]
    fn test_weighted_kmeans_negative_weights() {
        let data =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 1.2, 1.8, 4.0, 5.0, 4.2, 4.8]).unwrap();

        // Negative weights should cause an error
        let weights = Array1::from_vec(vec![1.0, -1.0, 1.0, 1.0]);

        let result = weighted_kmeans(data.view(), weights.view(), 2, None);
        assert!(result.is_err());
    }
}
