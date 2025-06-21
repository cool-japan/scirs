//! Enhanced K-means clustering implementation with multiple initialization methods

use ndarray::{s, Array1, Array2, ArrayView2};
use num_traits::{Float, FromPrimitive};
use rand::{rng, rngs::StdRng, Rng, RngCore, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::fmt::Debug;
use std::str::FromStr;

use super::{euclidean_distance, vq};
use crate::error::{ClusteringError, Result};
use scirs2_core::validation::{clustering::*, parameters::*};

/// Initialization methods for kmeans2
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MinitMethod {
    /// Generate k centroids from a Gaussian with mean and variance estimated from the data
    Random,
    /// Choose k observations (rows) at random from data for the initial centroids
    Points,
    /// K-means++ initialization (careful seeding)
    PlusPlus,
}

impl MinitMethod {
    /// Parse a string into a MinitMethod (SciPy-compatible)
    ///
    /// # Arguments
    ///
    /// * `s` - String representation of the initialization method
    ///
    /// # Returns
    ///
    /// * The corresponding MinitMethod enum value
    ///
    /// # Errors
    ///
    /// * Returns an error if the string is not recognized
    pub fn parse_method(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "random" => Ok(MinitMethod::Random),
            "points" => Ok(MinitMethod::Points),
            "k-means++" | "kmeans++" | "plusplus" => Ok(MinitMethod::PlusPlus),
            _ => Err(ClusteringError::InvalidInput(format!(
                "Unknown initialization method: '{}'. Valid options are: 'random', 'points', 'k-means++'",
                s
            ))),
        }
    }
}

impl FromStr for MinitMethod {
    type Err = ClusteringError;

    fn from_str(s: &str) -> Result<Self> {
        Self::parse_method(s)
    }
}

/// Methods for handling empty clusters
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingMethod {
    /// Give a warning and continue
    Warn,
    /// Raise a ClusteringError and terminate the algorithm
    Raise,
}

/// Enhanced K-means clustering algorithm compatible with SciPy's kmeans2
///
/// # Arguments
///
/// * `data` - Input data (n_samples × n_features) or pre-computed centroids if minit is None
/// * `k` - Number of clusters or initial centroids array
/// * `iter` - Number of iterations
/// * `thresh` - Convergence threshold (not used yet)
/// * `minit` - Initialization method (None if k is a centroid array)
/// * `missing` - Method to handle empty clusters
/// * `check_finite` - Whether to check input validity
/// * `random_seed` - Optional random seed
///
/// # Returns
///
/// * Tuple of (centroids, labels) where:
///   - centroids: Array of shape (k × n_features)
///   - labels: Array of shape (n_samples,) with cluster assignments
#[allow(clippy::too_many_arguments)]
pub fn kmeans2<F>(
    data: ArrayView2<F>,
    k: usize,
    iter: Option<usize>,
    thresh: Option<F>,
    minit: Option<MinitMethod>,
    missing: Option<MissingMethod>,
    check_finite: Option<bool>,
    random_seed: Option<u64>,
) -> Result<(Array2<F>, Array1<usize>)>
where
    F: Float + FromPrimitive + Debug + std::iter::Sum + std::fmt::Display,
{
    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];
    let iterations = iter.unwrap_or(10);
    let threshold = thresh.unwrap_or(F::from(1e-5).unwrap());
    let missing_method = missing.unwrap_or(MissingMethod::Warn);
    let check_finite_flag = check_finite.unwrap_or(true);

    // Use unified validation
    validate_clustering_data(&data, "K-means", check_finite_flag, Some(k))
        .map_err(|e| ClusteringError::InvalidInput(format!("K-means: {}", e)))?;

    check_n_clusters_bounds(&data, k, "K-means")
        .map_err(|e| ClusteringError::InvalidInput(format!("{}", e)))?;

    check_iteration_params(iterations, threshold, "K-means")
        .map_err(|e| ClusteringError::InvalidInput(format!("{}", e)))?;

    // Initialize centroids
    let init_method = minit.unwrap_or(MinitMethod::PlusPlus); // Default to k-means++
    let mut centroids = match init_method {
        MinitMethod::Random => krandinit(data, k, random_seed)?,
        MinitMethod::Points => kpoints(data, k, random_seed)?,
        MinitMethod::PlusPlus => kmeans_plus_plus(data, k, random_seed)?,
    };

    let mut labels;

    // Run K-means iterations
    for _iteration in 0..iterations {
        // Store previous centroids for convergence check
        let prev_centroids = centroids.clone();

        // Assign samples to nearest centroid
        let (new_labels, _distances) = vq(data, centroids.view())?;
        labels = new_labels;

        // Compute new centroids
        let mut new_centroids = Array2::zeros((k, n_features));
        let mut counts = Array1::zeros(k);

        for i in 0..n_samples {
            let cluster = labels[i];
            let point = data.slice(s![i, ..]);

            for j in 0..n_features {
                new_centroids[[cluster, j]] = new_centroids[[cluster, j]] + point[j];
            }

            counts[cluster] += 1;
        }

        // Handle empty clusters
        for i in 0..k {
            if counts[i] == 0 {
                match missing_method {
                    MissingMethod::Warn => {
                        eprintln!("One of the clusters is empty. Re-run kmeans with a different initialization.");
                        // Find point furthest from its centroid
                        let mut max_dist = F::zero();
                        let mut far_idx = 0;

                        for j in 0..n_samples {
                            let cluster_j = labels[j];
                            let dist = euclidean_distance(
                                data.slice(s![j, ..]),
                                centroids.slice(s![cluster_j, ..]),
                            );
                            if dist > max_dist {
                                max_dist = dist;
                                far_idx = j;
                            }
                        }

                        // Move this point to the empty cluster
                        for j in 0..n_features {
                            new_centroids[[i, j]] = data[[far_idx, j]];
                        }
                        counts[i] = 1;
                    }
                    MissingMethod::Raise => {
                        return Err(ClusteringError::EmptyCluster(
                            "One of the clusters is empty. Re-run kmeans with a different initialization.".to_string()
                        ));
                    }
                }
            } else {
                // Normalize by the number of points in the cluster
                for j in 0..n_features {
                    new_centroids[[i, j]] = new_centroids[[i, j]] / F::from(counts[i]).unwrap();
                }
            }
        }

        centroids = new_centroids;

        // Check for convergence
        let mut max_centroid_shift = F::zero();
        for i in 0..k {
            for j in 0..n_features {
                let shift = (centroids[[i, j]] - prev_centroids[[i, j]]).abs();
                if shift > max_centroid_shift {
                    max_centroid_shift = shift;
                }
            }
        }

        // If convergence reached, break early
        if max_centroid_shift < threshold {
            break;
        }
    }

    // Final assignment
    let (final_labels, _) = vq(data, centroids.view())?;

    Ok((centroids, final_labels))
}

/// SciPy-compatible K-means clustering with string-based parameters
///
/// This function provides an interface compatible with SciPy's kmeans2,
/// accepting string-based initialization methods.
///
/// # Arguments
///
/// * `data` - Input data (n_samples × n_features)
/// * `k` - Number of clusters
/// * `iter` - Number of iterations
/// * `thresh` - Convergence threshold
/// * `minit` - Initialization method as string ('random', 'points', 'k-means++')
/// * `missing` - Method to handle empty clusters ('warn', 'raise')
/// * `check_finite` - Whether to check input validity
/// * `random_seed` - Optional random seed
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
/// use ndarray::Array2;
/// use scirs2_cluster::vq::kmeans2_str;
///
/// let data = Array2::from_shape_vec((6, 2), vec![
///     1.0, 1.0, 1.1, 1.1, 0.9, 0.9,
///     8.0, 8.0, 8.1, 8.1, 7.9, 7.9,
/// ]).unwrap();
///
/// let (centroids, labels) = kmeans2_str(
///     data.view(), 2, Some(20), Some(1e-5), Some("k-means++"),
///     Some("warn"), Some(true), Some(42)
/// ).unwrap();
/// ```
#[allow(clippy::too_many_arguments)]
pub fn kmeans2_str<F>(
    data: ArrayView2<F>,
    k: usize,
    iter: Option<usize>,
    thresh: Option<F>,
    minit: Option<&str>,
    missing: Option<&str>,
    check_finite: Option<bool>,
    random_seed: Option<u64>,
) -> Result<(Array2<F>, Array1<usize>)>
where
    F: Float + FromPrimitive + Debug + std::iter::Sum + std::fmt::Display,
{
    // Parse string parameters
    let minit_method = if let Some(method_str) = minit {
        Some(MinitMethod::from_str(method_str)?)
    } else {
        Some(MinitMethod::PlusPlus) // Default to k-means++ like SciPy
    };

    let missing_method = if let Some(missing_str) = missing {
        match missing_str.to_lowercase().as_str() {
            "warn" => Some(MissingMethod::Warn),
            "raise" => Some(MissingMethod::Raise),
            _ => {
                return Err(ClusteringError::InvalidInput(format!(
                    "Unknown missing method: '{}'. Valid options are: 'warn', 'raise'",
                    missing_str
                )))
            }
        }
    } else {
        Some(MissingMethod::Warn) // Default to warn like SciPy
    };

    // Call the main kmeans2 function
    kmeans2(
        data,
        k,
        iter,
        thresh,
        minit_method,
        missing_method,
        check_finite,
        random_seed,
    )
}

/// Random initialization: generate k centroids from a Gaussian with mean and
/// variance estimated from the data
fn krandinit<F>(data: ArrayView2<F>, k: usize, random_seed: Option<u64>) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + std::iter::Sum,
{
    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];

    // Calculate mean and variance for each feature
    let mut means = Array1::<F>::zeros(n_features);
    let mut vars = Array1::<F>::zeros(n_features);

    for j in 0..n_features {
        let mut sum = F::zero();
        for i in 0..n_samples {
            sum = sum + data[[i, j]];
        }
        means[j] = sum / F::from(n_samples).unwrap();

        let mut var_sum = F::zero();
        for i in 0..n_samples {
            let diff = data[[i, j]] - means[j];
            var_sum = var_sum + diff * diff;
        }
        vars[j] = var_sum / F::from(n_samples).unwrap();
    }

    // Generate random centroids from Gaussian distribution
    let mut centroids = Array2::<F>::zeros((k, n_features));

    let mut rng: Box<dyn RngCore> = if let Some(seed) = random_seed {
        Box::new(StdRng::from_seed([seed as u8; 32]))
    } else {
        Box::new(rng())
    };

    for i in 0..k {
        for j in 0..n_features {
            // Convert to f64 for normal distribution
            let mean = means[j].to_f64().unwrap();
            let std = vars[j].sqrt().to_f64().unwrap();

            if std > 0.0 {
                let normal = Normal::new(mean, std).unwrap();
                let value = normal.sample(&mut rng);
                centroids[[i, j]] = F::from(value).unwrap();
            } else {
                centroids[[i, j]] = means[j];
            }
        }
    }

    Ok(centroids)
}

/// Points initialization: choose k observations (rows) at random from data
fn kpoints<F>(data: ArrayView2<F>, k: usize, random_seed: Option<u64>) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];

    let mut rng: Box<dyn RngCore> = if let Some(seed) = random_seed {
        Box::new(StdRng::from_seed([seed as u8; 32]))
    } else {
        Box::new(rng())
    };

    // Choose k random indices without replacement
    let mut indices: Vec<usize> = (0..n_samples).collect();

    // Shuffle and take first k
    for i in 0..k {
        let j = rng.random_range(i..n_samples);
        indices.swap(i, j);
    }

    // Extract centroids from data
    let mut centroids = Array2::zeros((k, n_features));
    for i in 0..k {
        let idx = indices[i];
        for j in 0..n_features {
            centroids[[i, j]] = data[[idx, j]];
        }
    }

    Ok(centroids)
}

/// K-means++ initialization
fn kmeans_plus_plus<F>(data: ArrayView2<F>, k: usize, random_seed: Option<u64>) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + std::iter::Sum,
{
    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];

    let mut rng: Box<dyn RngCore> = if let Some(seed) = random_seed {
        Box::new(StdRng::from_seed([seed as u8; 32]))
    } else {
        Box::new(rng())
    };

    let mut centroids = Array2::zeros((k, n_features));

    // Choose first centroid randomly
    let first_idx = rng.random_range(0..n_samples);
    for j in 0..n_features {
        centroids[[0, j]] = data[[first_idx, j]];
    }

    // Choose remaining centroids
    for i in 1..k {
        // Calculate squared distances to nearest centroid
        let mut distances = Array1::<F>::zeros(n_samples);

        for j in 0..n_samples {
            let mut min_dist = F::infinity();
            for c in 0..i {
                let dist = euclidean_distance(data.slice(s![j, ..]), centroids.slice(s![c, ..]));
                if dist < min_dist {
                    min_dist = dist;
                }
            }
            distances[j] = min_dist * min_dist;
        }

        // Calculate probabilities
        let total = distances.iter().fold(F::zero(), |a, &b| a + b);
        let mut probabilities = Array1::<F>::zeros(n_samples);
        for j in 0..n_samples {
            probabilities[j] = distances[j] / total;
        }

        // Choose next centroid based on weighted probability
        let mut cumsum = F::zero();
        let r = F::from(rng.random::<f64>()).unwrap();
        let mut next_idx = n_samples - 1;

        for j in 0..n_samples {
            cumsum = cumsum + probabilities[j];
            if cumsum > r {
                next_idx = j;
                break;
            }
        }

        // Add chosen centroid
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
    use ndarray::{array, Array2};

    #[test]
    fn test_kmeans2_basic_functionality() {
        let data = array![
            [1.0, 1.0],
            [1.5, 1.5],
            [0.8, 0.9],
            [8.0, 8.0],
            [8.2, 8.1],
            [7.8, 7.9],
        ];

        let (centroids, labels) = kmeans2(
            data.view(),
            2,
            Some(50),
            Some(1e-6),
            Some(MinitMethod::PlusPlus),
            Some(MissingMethod::Warn),
            Some(true),
            Some(42),
        )
        .unwrap();

        // Should have 2 centroids
        assert_eq!(centroids.shape(), [2, 2]);

        // Should have labels for all 6 points
        assert_eq!(labels.len(), 6);

        // All labels should be 0 or 1
        assert!(labels.iter().all(|&l| l == 0 || l == 1));

        // Should have points from both clusters
        let unique_labels: std::collections::HashSet<_> = labels.iter().cloned().collect();
        assert_eq!(unique_labels.len(), 2);
    }

    #[test]
    fn test_kmeans2_parameter_validation() {
        let data = array![[1.0, 1.0], [2.0, 2.0]];

        // Test k=0 (invalid)
        let result = kmeans2(
            data.view(),
            0,
            None,
            None,
            Some(MinitMethod::Random),
            None,
            None,
            None,
        );
        assert!(result.is_err());

        // Test k > n_samples (invalid)
        let result = kmeans2(
            data.view(),
            5,
            None,
            None,
            Some(MinitMethod::Random),
            None,
            None,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_kmeans2_initialization_methods() {
        let data = array![
            [1.0, 1.0],
            [1.5, 1.5],
            [0.8, 0.9],
            [8.0, 8.0],
            [8.2, 8.1],
            [7.8, 7.9],
        ];

        let methods = vec![
            MinitMethod::Random,
            MinitMethod::Points,
            MinitMethod::PlusPlus,
        ];

        for method in methods {
            let result = kmeans2(
                data.view(),
                2,
                Some(10),
                None,
                Some(method),
                Some(MissingMethod::Warn),
                None,
                Some(42),
            );

            assert!(result.is_ok(), "Failed with method: {:?}", method);
            let (centroids, labels) = result.unwrap();
            assert_eq!(centroids.shape(), [2, 2]);
            assert_eq!(labels.len(), 6);
        }
    }

    #[test]
    fn test_kmeans2_reproducibility_with_seed() {
        let data = array![
            [1.0, 1.0],
            [1.5, 1.5],
            [0.8, 0.9],
            [8.0, 8.0],
            [8.2, 8.1],
            [7.8, 7.9],
        ];

        let (centroids1, labels1) = kmeans2(
            data.view(),
            2,
            Some(10),
            None,
            Some(MinitMethod::Random),
            None,
            None,
            Some(42),
        )
        .unwrap();

        let (centroids2, labels2) = kmeans2(
            data.view(),
            2,
            Some(10),
            None,
            Some(MinitMethod::Random),
            None,
            None,
            Some(42),
        )
        .unwrap();

        // With same seed, results should be identical
        assert_eq!(labels1, labels2);

        // Centroids should be very close (allowing for floating point precision)
        for i in 0..centroids1.shape()[0] {
            for j in 0..centroids1.shape()[1] {
                assert_abs_diff_eq!(centroids1[[i, j]], centroids2[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_kmeans2_single_cluster() {
        let data = array![[1.0, 1.0], [1.1, 1.1], [0.9, 0.9],];

        let (centroids, labels) = kmeans2(
            data.view(),
            1,
            Some(10),
            None,
            Some(MinitMethod::Points),
            None,
            None,
            Some(42),
        )
        .unwrap();

        // Should have 1 centroid
        assert_eq!(centroids.shape(), [1, 2]);

        // All labels should be 0
        assert!(labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn test_kmeans2_identical_points() {
        let data = array![[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0],];

        let (centroids, labels) = kmeans2(
            data.view(),
            2,
            Some(10),
            None,
            Some(MinitMethod::Points),
            Some(MissingMethod::Warn),
            None,
            Some(42),
        )
        .unwrap();

        // Should still produce valid results
        assert_eq!(centroids.shape(), [2, 2]);
        assert_eq!(labels.len(), 4);

        // All labels should be valid (0 or 1)
        assert!(labels.iter().all(|&l| l == 0 || l == 1));
    }

    #[test]
    fn test_kmeans2_missing_method_warn() {
        // Create data that might lead to empty clusters
        let data = array![[0.0, 0.0], [0.1, 0.1], [10.0, 10.0],];

        let result = kmeans2(
            data.view(),
            2,
            Some(5),
            None,
            Some(MinitMethod::Random),
            Some(MissingMethod::Warn),
            None,
            Some(123),
        );

        // Should succeed with warning
        assert!(result.is_ok());
    }

    #[test]
    fn test_kmeans2_convergence_behavior() {
        // Create well-separated clusters
        let data = array![
            [1.0, 1.0],
            [1.1, 1.1],
            [0.9, 0.9],
            [10.0, 10.0],
            [10.1, 10.1],
            [9.9, 9.9],
        ];

        // Test with different iteration counts
        let (centroids_few, _) = kmeans2(
            data.view(),
            2,
            Some(1),
            None,
            Some(MinitMethod::PlusPlus),
            None,
            None,
            Some(42),
        )
        .unwrap();

        let (centroids_many, _) = kmeans2(
            data.view(),
            2,
            Some(100),
            None,
            Some(MinitMethod::PlusPlus),
            None,
            None,
            Some(42),
        )
        .unwrap();

        // Results should be valid for both
        assert_eq!(centroids_few.shape(), [2, 2]);
        assert_eq!(centroids_many.shape(), [2, 2]);
    }

    #[test]
    fn test_kmeans2_high_k() {
        let data = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0],];

        // Test with k equal to number of points
        let (centroids, labels) = kmeans2(
            data.view(),
            5,
            Some(10),
            None,
            Some(MinitMethod::Points),
            None,
            None,
            Some(42),
        )
        .unwrap();

        assert_eq!(centroids.shape(), [5, 2]);
        assert_eq!(labels.len(), 5);

        // Each point should be its own cluster
        let unique_labels: std::collections::HashSet<_> = labels.iter().cloned().collect();
        assert_eq!(unique_labels.len(), 5);
    }

    #[test]
    fn test_kmeans2_different_thresholds() {
        let data = array![[1.0, 1.0], [1.5, 1.5], [8.0, 8.0], [8.5, 8.5],];

        // Test with different convergence thresholds
        let result1 = kmeans2(
            data.view(),
            2,
            Some(100),
            Some(1e-10), // Very strict
            Some(MinitMethod::PlusPlus),
            None,
            None,
            Some(42),
        );

        let result2 = kmeans2(
            data.view(),
            2,
            Some(100),
            Some(1e-1), // Very loose
            Some(MinitMethod::PlusPlus),
            None,
            None,
            Some(42),
        );

        assert!(result1.is_ok());
        assert!(result2.is_ok());
    }

    #[test]
    fn test_kmeans2_convergence_threshold() {
        // Test early convergence with well-separated clusters
        let data = array![
            [1.0, 1.0],
            [1.1, 1.1],
            [0.9, 0.9],
            [10.0, 10.0],
            [10.1, 10.1],
            [9.9, 9.9],
        ];

        // With a tight threshold, should converge quickly
        let result1 = kmeans2(
            data.view(),
            2,
            Some(100),   // Allow many iterations
            Some(1e-10), // Very strict threshold - should converge quickly
            Some(MinitMethod::PlusPlus),
            None,
            None,
            Some(42),
        );

        assert!(result1.is_ok());
        let (centroids1, labels1) = result1.unwrap();
        assert_eq!(centroids1.shape(), [2, 2]);
        assert_eq!(labels1.len(), 6);

        // With a loose threshold, should also work
        let result2 = kmeans2(
            data.view(),
            2,
            Some(100),
            Some(1e-1), // Very loose threshold
            Some(MinitMethod::PlusPlus),
            None,
            None,
            Some(42),
        );

        assert!(result2.is_ok());
        let (centroids2, labels2) = result2.unwrap();
        assert_eq!(centroids2.shape(), [2, 2]);
        assert_eq!(labels2.len(), 6);
    }

    #[test]
    fn test_kmeans2_check_finite() {
        // Test with finite data (should work)
        let data = array![[1.0, 2.0], [1.5, 1.5], [8.0, 8.0],];

        let result = kmeans2(
            data.view(),
            2,
            Some(10),
            None,
            Some(MinitMethod::Random),
            None,
            Some(true), // check_finite = true
            Some(42),
        );
        assert!(result.is_ok());

        // Test with check_finite disabled (should also work with finite data)
        let result = kmeans2(
            data.view(),
            2,
            Some(10),
            None,
            Some(MinitMethod::Random),
            None,
            Some(false), // check_finite = false
            Some(42),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_kmeans2_large_dataset() {
        // Generate a larger dataset for stress testing
        let mut data = Array2::zeros((100, 3));

        // Create 3 clusters
        for i in 0..100 {
            let cluster = i % 3;
            match cluster {
                0 => {
                    data[[i, 0]] = 1.0 + (i as f64) * 0.01;
                    data[[i, 1]] = 1.0 + (i as f64) * 0.01;
                    data[[i, 2]] = 1.0 + (i as f64) * 0.01;
                }
                1 => {
                    data[[i, 0]] = 5.0 + (i as f64) * 0.01;
                    data[[i, 1]] = 5.0 + (i as f64) * 0.01;
                    data[[i, 2]] = 5.0 + (i as f64) * 0.01;
                }
                2 => {
                    data[[i, 0]] = 10.0 + (i as f64) * 0.01;
                    data[[i, 1]] = 10.0 + (i as f64) * 0.01;
                    data[[i, 2]] = 10.0 + (i as f64) * 0.01;
                }
                _ => unreachable!(),
            }
        }

        let (centroids, labels) = kmeans2(
            data.view(),
            3,
            Some(50),
            None,
            Some(MinitMethod::PlusPlus),
            None,
            None,
            Some(42),
        )
        .unwrap();

        assert_eq!(centroids.shape(), [3, 3]);
        assert_eq!(labels.len(), 100);

        // Should find 3 clusters
        let unique_labels: std::collections::HashSet<_> = labels.iter().cloned().collect();
        assert_eq!(unique_labels.len(), 3);
    }

    // Tests for string-based parameter support
    use super::kmeans2_str;

    #[test]
    fn test_kmeans2_str_basic_functionality() {
        let data = array![
            [1.0, 1.0],
            [1.5, 1.5],
            [0.8, 0.9],
            [8.0, 8.0],
            [8.2, 8.1],
            [7.8, 7.9],
        ];

        let (centroids, labels) = kmeans2_str(
            data.view(),
            2,
            Some(50),
            Some(1e-6),
            Some("k-means++"),
            Some("warn"),
            Some(true),
            Some(42),
        )
        .unwrap();

        assert_eq!(centroids.shape(), [2, 2]);
        assert_eq!(labels.len(), 6);
        assert!(labels.iter().all(|&l| l == 0 || l == 1));

        let unique_labels: std::collections::HashSet<_> = labels.iter().cloned().collect();
        assert_eq!(unique_labels.len(), 2);
    }

    #[test]
    fn test_kmeans2_str_all_init_methods() {
        let data = array![
            [1.0, 1.0],
            [1.5, 1.5],
            [0.8, 0.9],
            [8.0, 8.0],
            [8.2, 8.1],
            [7.8, 7.9],
        ];

        let methods = vec!["random", "points", "k-means++", "kmeans++", "plusplus"];

        for method in methods {
            let result = kmeans2_str(
                data.view(),
                2,
                Some(10),
                None,
                Some(method),
                Some("warn"),
                None,
                Some(42),
            );

            assert!(result.is_ok(), "Failed with method: '{}'", method);
            let (centroids, labels) = result.unwrap();
            assert_eq!(centroids.shape(), [2, 2]);
            assert_eq!(labels.len(), 6);
        }
    }

    #[test]
    fn test_kmeans2_str_case_insensitive() {
        let data = array![[1.0, 1.0], [2.0, 2.0], [8.0, 8.0], [9.0, 9.0],];

        // Test case insensitive method names
        let methods = vec![
            "RANDOM",
            "Random",
            "random",
            "POINTS",
            "Points",
            "points",
            "K-MEANS++",
            "K-Means++",
            "k-means++",
        ];

        for method in methods {
            let result = kmeans2_str(
                data.view(),
                2,
                Some(10),
                None,
                Some(method),
                Some("warn"),
                None,
                Some(42),
            );

            assert!(result.is_ok(), "Failed with method: '{}'", method);
        }
    }

    #[test]
    fn test_kmeans2_str_missing_methods() {
        let data = array![[1.0, 1.0], [2.0, 2.0], [8.0, 8.0],];

        // Test different missing methods
        let missing_methods = vec!["warn", "raise", "WARN", "RAISE"];

        for missing_method in missing_methods {
            let result = kmeans2_str(
                data.view(),
                2,
                Some(5),
                None,
                Some("points"),
                Some(missing_method),
                None,
                Some(42),
            );

            assert!(
                result.is_ok(),
                "Failed with missing method: '{}'",
                missing_method
            );
        }
    }

    #[test]
    fn test_kmeans2_str_invalid_method() {
        let data = array![[1.0, 1.0], [2.0, 2.0]];

        // Test invalid initialization method
        let result = kmeans2_str(
            data.view(),
            2,
            Some(10),
            None,
            Some("invalid_method"),
            Some("warn"),
            None,
            None,
        );

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Unknown initialization method"));
    }

    #[test]
    fn test_kmeans2_str_invalid_missing_method() {
        let data = array![[1.0, 1.0], [2.0, 2.0]];

        // Test invalid missing method
        let result = kmeans2_str(
            data.view(),
            2,
            Some(10),
            None,
            Some("points"),
            Some("invalid_missing"),
            None,
            None,
        );

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Unknown missing method"));
    }

    #[test]
    fn test_kmeans2_str_defaults() {
        let data = array![[1.0, 1.0], [1.5, 1.5], [8.0, 8.0], [8.5, 8.5],];

        // Test with all None parameters (should use defaults)
        let result = kmeans2_str(
            data.view(),
            2,
            Some(10),
            None,
            None, // Should default to k-means++
            None, // Should default to warn
            None,
            Some(42),
        );

        assert!(result.is_ok());
        let (centroids, labels) = result.unwrap();
        assert_eq!(centroids.shape(), [2, 2]);
        assert_eq!(labels.len(), 4);
    }

    #[test]
    fn test_kmeans2_str_equivalence_with_enum() {
        let data = array![
            [1.0, 1.0],
            [1.5, 1.5],
            [0.8, 0.9],
            [8.0, 8.0],
            [8.2, 8.1],
            [7.8, 7.9],
        ];

        // Test that string version produces same results as enum version
        let (centroids_enum, labels_enum) = kmeans2(
            data.view(),
            2,
            Some(50),
            Some(1e-6),
            Some(MinitMethod::PlusPlus),
            Some(MissingMethod::Warn),
            Some(true),
            Some(42),
        )
        .unwrap();

        let (centroids_str, labels_str) = kmeans2_str(
            data.view(),
            2,
            Some(50),
            Some(1e-6),
            Some("k-means++"),
            Some("warn"),
            Some(true),
            Some(42),
        )
        .unwrap();

        // Results should be identical
        assert_eq!(labels_enum, labels_str);

        for i in 0..centroids_enum.shape()[0] {
            for j in 0..centroids_enum.shape()[1] {
                assert_abs_diff_eq!(
                    centroids_enum[[i, j]],
                    centroids_str[[i, j]],
                    epsilon = 1e-10
                );
            }
        }
    }

    #[test]
    fn test_minit_method_from_str() {
        // Test MinitMethod::from_str function directly
        assert_eq!(
            MinitMethod::from_str("random").unwrap(),
            MinitMethod::Random
        );
        assert_eq!(
            MinitMethod::from_str("RANDOM").unwrap(),
            MinitMethod::Random
        );
        assert_eq!(
            MinitMethod::from_str("points").unwrap(),
            MinitMethod::Points
        );
        assert_eq!(
            MinitMethod::from_str("POINTS").unwrap(),
            MinitMethod::Points
        );
        assert_eq!(
            MinitMethod::from_str("k-means++").unwrap(),
            MinitMethod::PlusPlus
        );
        assert_eq!(
            MinitMethod::from_str("kmeans++").unwrap(),
            MinitMethod::PlusPlus
        );
        assert_eq!(
            MinitMethod::from_str("plusplus").unwrap(),
            MinitMethod::PlusPlus
        );
        assert_eq!(
            MinitMethod::from_str("K-MEANS++").unwrap(),
            MinitMethod::PlusPlus
        );

        // Test invalid method
        assert!(MinitMethod::from_str("invalid").is_err());
    }
}
