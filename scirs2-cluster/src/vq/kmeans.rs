//! K-means clustering implementation

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, FromPrimitive};
use rand::Rng;
use std::fmt::Debug;

use super::{euclidean_distance, vq};
use crate::error::{ClusteringError, Result};
// use scirs2_core::validation::{clustering::*, parameters::*};

// Re-export kmeans2 related types and functions

/// Options for K-means clustering
#[derive(Debug, Clone)]
pub struct KMeansOptions<F: Float> {
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

impl<F: Float + FromPrimitive> Default for KMeansOptions<F> {
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

/// K-means clustering algorithm (SciPy-compatible version)
///
/// # Arguments
///
/// * `obs` - Input data (n_samples × n_features)
/// * `k_or_guess` - Number of clusters or initial guess for centroids
/// * `iter` - Maximum number of iterations (default: 20)
/// * `thresh` - Convergence threshold (default: 1e-5)
/// * `check_finite` - Whether to check for finite values (default: true)
/// * `seed` - Random seed for initialization (optional)
///
/// # Returns
///
/// * Tuple of (centroids, distortion) where:
///   - centroids: Array of shape (k × n_features)
///   - distortion: Sum of squared distances to centroids
///
/// # Examples
///
/// ```
/// use ndarray::{ArrayView1, Array2, ArrayView2};
/// use scirs2_cluster::vq::kmeans;
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
/// let (centroids, distortion) = kmeans(data.view(), 2, Some(20), Some(1e-5), Some(true), Some(42)).unwrap();
/// ```
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn kmeans<F>(
    obs: ArrayView2<F>,
    k_or_guess: usize,
    iter: Option<usize>,
    thresh: Option<F>,
    check_finite: Option<bool>,
    seed: Option<u64>,
) -> Result<(Array2<F>, F)>
where
    F: Float + FromPrimitive + Debug + std::iter::Sum + std::fmt::Display,
{
    let k = k_or_guess; // For now, just treat as number of clusters
    let max_iter = iter.unwrap_or(20);
    let tol = thresh.unwrap_or(F::from(1e-5).unwrap());
    let _check_finite_flag = check_finite.unwrap_or(true);

    // Basic validation
    if obs.is_empty() {
        return Err(ClusteringError::InvalidInput(
            "Input data is empty".to_string(),
        ));
    }
    if k == 0 {
        return Err(ClusteringError::InvalidInput(
            "Number of clusters must be greater than 0".to_string(),
        ));
    }
    if k > obs.nrows() {
        return Err(ClusteringError::InvalidInput(format!(
            "Number of clusters ({}) cannot be greater than number of data points ({})",
            k,
            obs.nrows()
        )));
    }

    // Create options struct for internal use
    let options = KMeansOptions {
        max_iter,
        tol,
        random_seed: seed,
        n_init: 1, // SciPy's kmeans does single initialization
        init_method: KMeansInit::KMeansPlusPlus,
    };

    // Use the options-based version internally
    let (centroids, labels) = kmeans_with_options(obs, k, Some(options))?;

    // Calculate distortion (sum of squared distances to centroids)
    let distortion = calculate_distortion(obs, centroids.view(), &labels);

    Ok((centroids, distortion))
}

/// K-means clustering algorithm (options-based version)
///
/// This is the original implementation that uses the options struct.
/// The SciPy-compatible version above is a wrapper around this function.
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
/// use ndarray::{ArrayView1, Array2, ArrayView2};
/// use scirs2_cluster::vq::kmeans_with_options;
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
/// let (centroids, labels) = kmeans_with_options(data.view(), 2, None).unwrap();
/// ```
#[allow(dead_code)]
pub fn kmeans_with_options<F>(
    data: ArrayView2<F>,
    k: usize,
    options: Option<KMeansOptions<F>>,
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

    if k > n_samples {
        return Err(ClusteringError::InvalidInput(format!(
            "Number of clusters ({}) cannot be greater than number of data points ({})",
            k, n_samples
        )));
    }

    let opts = options.unwrap_or_default();
    // Random seed is handled in kmeans_init function

    let mut bestcentroids = None;
    let mut best_labels = None;
    let mut best_inertia = F::infinity();

    // If we're using K-means|| initialization, we only need to run once
    let n_init = if opts.init_method == KMeansInit::KMeansParallel {
        1
    } else {
        opts.n_init
    };

    for _ in 0..n_init {
        // Initialize centroids using the specified method
        let centroids = kmeans_init(data, k, Some(opts.init_method), opts.random_seed)?;

        // Run k-means
        let (centroids, labels, inertia) = _kmeans_single(data, centroids.view(), &opts)?;

        if inertia < best_inertia {
            bestcentroids = Some(centroids);
            best_labels = Some(labels);
            best_inertia = inertia;
        }
    }

    Ok((bestcentroids.unwrap(), best_labels.unwrap()))
}

/// Calculate distortion (sum of squared distances to centroids)
#[allow(dead_code)]
fn calculate_distortion<F>(
    data: ArrayView2<F>,
    centroids: ArrayView2<F>,
    labels: &Array1<usize>,
) -> F
where
    F: Float + FromPrimitive + Debug + std::iter::Sum,
{
    let n_samples = data.shape()[0];
    let mut total_distortion = F::zero();

    for i in 0..n_samples {
        let cluster = labels[i];
        let point = data.slice(s![i, ..]);
        let centroid = centroids.slice(s![cluster, ..]);

        let squared_distance = euclidean_distance(point, centroid).powi(2);
        total_distortion = total_distortion + squared_distance;
    }

    total_distortion
}

/// Run a single k-means clustering iteration
#[allow(dead_code)]
fn _kmeans_single<F>(
    data: ArrayView2<F>,
    initcentroids: ArrayView2<F>,
    opts: &KMeansOptions<F>,
) -> Result<(Array2<F>, Array1<usize>, F)>
where
    F: Float + FromPrimitive + Debug + std::iter::Sum,
{
    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];
    let k = initcentroids.shape()[0];

    let mut centroids = initcentroids.to_owned();
    let mut labels = Array1::zeros(n_samples);
    let mut prev_centroid_diff = F::infinity();

    for _iter in 0..opts.max_iter {
        // Assign samples to nearest centroid
        let (new_labels, distances) = vq(data, centroids.view())?;
        labels = new_labels;

        // Compute new centroids
        let mut newcentroids = Array2::zeros((k, n_features));
        let mut counts = Array1::zeros(k);

        for i in 0..n_samples {
            let cluster = labels[i];
            let point = data.slice(s![i, ..]);

            for j in 0..n_features {
                newcentroids[[cluster, j]] = newcentroids[[cluster, j]] + point[j];
            }

            counts[cluster] += 1;
        }

        // If a cluster is empty, reinitialize it
        for i in 0..k {
            if counts[i] == 0 {
                // Find the point furthest from its centroid
                let mut max_dist = F::zero();
                let mut far_idx = 0;

                for j in 0..n_samples {
                    let dist = distances[j];
                    if dist > max_dist {
                        max_dist = dist;
                        far_idx = j;
                    }
                }

                // Move this point to the empty cluster
                for j in 0..n_features {
                    newcentroids[[i, j]] = data[[far_idx, j]];
                }

                counts[i] = 1;
            } else {
                // Normalize by the number of points in the cluster
                for j in 0..n_features {
                    newcentroids[[i, j]] = newcentroids[[i, j]] / F::from(counts[i]).unwrap();
                }
            }
        }

        // Check for convergence
        let mut centroid_diff = F::zero();
        for i in 0..k {
            let dist =
                euclidean_distance(centroids.slice(s![i, ..]), newcentroids.slice(s![i, ..]));
            centroid_diff = centroid_diff + dist;
        }

        centroids = newcentroids;

        if centroid_diff <= opts.tol || centroid_diff >= prev_centroid_diff {
            break;
        }

        prev_centroid_diff = centroid_diff;
    }

    // Calculate inertia (sum of squared distances to nearest centroid)
    let mut inertia = F::zero();
    for i in 0..n_samples {
        let cluster = labels[i];
        let dist = euclidean_distance(data.slice(s![i, ..]), centroids.slice(s![cluster, ..]));
        inertia = inertia + dist * dist;
    }

    Ok((centroids, labels, inertia))
}

/// Initialization methods for K-means
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KMeansInit {
    /// Random initialization
    Random,
    /// K-means++ initialization
    #[default]
    KMeansPlusPlus,
    /// K-means|| initialization (parallel version of K-means++)
    KMeansParallel,
}

/// K-means initialization algorithm
///
/// # Arguments
///
/// * `data` - Input data (n_samples × n_features)
/// * `k` - Number of clusters
/// * `init_method` - Initialization method (default: K-means++)
/// * `random_seed` - Optional random seed
///
/// # Returns
///
/// * Array of shape (k × n_features) with initial centroids
#[allow(dead_code)]
pub fn kmeans_init<F>(
    data: ArrayView2<F>,
    k: usize,
    init_method: Option<KMeansInit>,
    random_seed: Option<u64>,
) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + std::iter::Sum,
{
    match init_method.unwrap_or_default() {
        KMeansInit::Random => random_init(data, k, random_seed),
        KMeansInit::KMeansPlusPlus => kmeans_plus_plus(data, k, random_seed),
        KMeansInit::KMeansParallel => kmeans_parallel(data, k, random_seed),
    }
}

/// Random initialization algorithm for K-means
///
/// # Arguments
///
/// * `data` - Input data (n_samples × n_features)
/// * `k` - Number of clusters
/// * `random_seed` - Optional random seed
///
/// # Returns
///
/// * Array of shape (k × n_features) with initial centroids
#[allow(dead_code)]
pub fn random_init<F>(data: ArrayView2<F>, k: usize, random_seed: Option<u64>) -> Result<Array2<F>>
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

    let mut rng = rand::rng();
    let mut centroids = Array2::zeros((k, n_features));
    let mut selected_indices = Vec::with_capacity(k);

    // Select k unique random points from the _data
    while selected_indices.len() < k {
        let idx = rng.random_range(0..n_samples);
        if !selected_indices.contains(&idx) {
            selected_indices.push(idx);
        }
    }

    // Copy the selected points to the centroids
    for (i, &idx) in selected_indices.iter().enumerate() {
        for j in 0..n_features {
            centroids[[i, j]] = data[[idx, j]];
        }
    }

    Ok(centroids)
}

/// K-means++ initialization algorithm
///
/// # Arguments
///
/// * `data` - Input data (n_samples × n_features)
/// * `k` - Number of clusters
/// * `_random_seed` - Optional random seed
///
/// # Returns
///
/// * Array of shape (k × n_features) with initial centroids
#[allow(dead_code)]
pub fn kmeans_plus_plus<F>(
    data: ArrayView2<F>,
    k: usize,
    random_seed: Option<u64>,
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

    let mut rng = rand::rng();

    let mut centroids = Array2::zeros((k, n_features));

    // Choose the first centroid randomly
    let first_idx = rng.random_range(0..n_samples);
    for j in 0..n_features {
        centroids[[0, j]] = data[[first_idx, j]];
    }

    if k == 1 {
        return Ok(centroids);
    }

    // Choose remaining centroids using the k-means++ algorithm
    for i in 1..k {
        // Compute distances to closest centroid for each point
        let mut min_distances = Array1::from_elem(n_samples, F::infinity());

        for sample_idx in 0..n_samples {
            let sample = data.slice(s![sample_idx, ..]);

            for centroid_idx in 0..i {
                let centroid = centroids.slice(s![centroid_idx, ..]);
                let dist = euclidean_distance(sample, centroid);

                if dist < min_distances[sample_idx] {
                    min_distances[sample_idx] = dist;
                }
            }
        }

        // Square the distances to get the probability distribution
        let mut weights = min_distances.mapv(|d| d * d);

        // Normalize the weights to create a probability distribution
        let sum_weights = weights.sum();
        if sum_weights > F::zero() {
            weights.mapv_inplace(|w| w / sum_weights);
        } else {
            // If all weights are zero, use uniform distribution
            weights.fill(F::from(1.0 / n_samples as f64).unwrap());
        }

        // Convert weights to cumulative distribution
        let mut cum_weights = weights.clone();
        for j in 1..n_samples {
            cum_weights[j] = cum_weights[j] + cum_weights[j - 1];
        }

        // Sample the next centroid based on the probability distribution
        let rand_val = F::from(rng.random_range(0.0..1.0)).unwrap();
        let mut next_idx = 0;

        for j in 0..n_samples {
            if rand_val <= cum_weights[j] {
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

/// K-means|| initialization algorithm (parallel version of K-means++)
///
/// This algorithm samples more than one center at each step, which makes it
/// suitable for parallel or distributed implementations.
///
/// # Arguments
///
/// * `data` - Input data (n_samples × n_features)
/// * `k` - Number of clusters
/// * `_random_seed` - Optional random seed
///
/// # Returns
///
/// * Array of shape (k × n_features) with initial centroids
///
/// # References
///
/// * [Scalable K-means++ by Bahmani et al.](https://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf)
#[allow(dead_code)]
pub fn kmeans_parallel<F>(
    data: ArrayView2<F>,
    k: usize,
    random_seed: Option<u64>,
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

    let mut rng = rand::rng();

    // Hyperparameters for K-means||
    let l = F::from(5.0).unwrap(); // Multiplication factor for oversampling
    let n_rounds = 8; // Number of rounds for parallel sampling

    // Centers is a weighted set of candidate centers
    let mut centers = Vec::new();
    let mut weights = Vec::new();

    // Choose the first center randomly
    let first_idx = rng.random_range(0..n_samples);
    let mut first_center = Vec::with_capacity(n_features);
    for j in 0..n_features {
        first_center.push(data[[first_idx, j]]);
    }
    centers.push(first_center);
    weights.push(F::one()); // Initial weight is 1

    // Perform parallel sampling rounds
    for _ in 0..n_rounds {
        // Compute distances to the closest center for each point
        let mut min_distances = Array1::from_elem(n_samples, F::infinity());

        for sample_idx in 0..n_samples {
            let sample = data.slice(s![sample_idx, ..]);

            for center in centers.iter() {
                let mut dist_sq = F::zero();
                for j in 0..n_features {
                    let diff = sample[j] - center[j];
                    dist_sq = dist_sq + diff * diff;
                }
                let dist = dist_sq.sqrt();

                if dist < min_distances[sample_idx] {
                    min_distances[sample_idx] = dist;
                }
            }
        }

        // Compute the sum of squared minimum distances (a.k.a. potential)
        let potential: F = min_distances.iter().map(|&d| d * d).sum();
        if potential <= F::epsilon() {
            break; // Already covered all points well
        }

        // Sample new centers proportional to their squared distance
        let expected_new_centers = l * F::from(k).unwrap();
        let oversampling = F::min(expected_new_centers / potential, F::one());

        for sample_idx in 0..n_samples {
            let probability = min_distances[sample_idx] * min_distances[sample_idx] * oversampling;

            // Sample with probability proportional to distance^2
            if F::from(rng.random_range(0.0..1.0)).unwrap() < probability {
                let mut new_center = Vec::with_capacity(n_features);
                for j in 0..n_features {
                    new_center.push(data[[sample_idx, j]]);
                }
                centers.push(new_center);
                weights.push(F::one()); // Initial weight is 1
            }
        }
    }

    // If we have too many candidate centers, cluster them using weighted k-means
    match centers.len().cmp(&k) {
        std::cmp::Ordering::Greater => {
            // Convert centers and weights to arrays for clustering
            let n_centers = centers.len();
            let mut centers_array = Array2::zeros((n_centers, n_features));
            let mut weights_array = Array1::zeros(n_centers);

            for i in 0..n_centers {
                for j in 0..n_features {
                    centers_array[[i, j]] = centers[i][j];
                }
                weights_array[i] = weights[i];
            }

            // Use regular k-means with weights to reduce to k centers
            let options = KMeansOptions {
                max_iter: 100,
                tol: F::from(1e-4).unwrap(),
                random_seed,
                n_init: 1,
                init_method: KMeansInit::KMeansPlusPlus,
            };

            // Initialize with random k centers from the candidate centers
            let init_indices: Vec<usize> = (0..n_centers)
            .filter(|_| rng.random_range(0.0..1.0) < 0.5) // Randomly select some centers
            .take(k) // Take at most k centers
            .collect();

            // If we didn't get k centers..just take the first k
            let actual_indices = if init_indices.len() < k {
                (0..k.min(n_centers)).collect::<Vec<usize>>()
            } else {
                init_indices
            };

            let mut initcentroids = Array2::zeros((actual_indices.len(), n_features));
            for (i, &idx) in actual_indices.iter().enumerate() {
                for j in 0..n_features {
                    initcentroids[[i, j]] = centers_array[[idx, j]];
                }
            }

            // Run weighted k-means to get final centroids
            let (finalcentroids_, _) = _weighted_kmeans_single(
                centers_array.view(),
                weights_array.view(),
                initcentroids.view(),
                &options,
            )?;

            Ok(finalcentroids_)
        }
        std::cmp::Ordering::Less => {
            // If we have too few centers, add random points
            let mut centroids = Array2::zeros((k, n_features));

            // Copy existing centers
            for i in 0..centers.len() {
                for j in 0..n_features {
                    centroids[[i, j]] = centers[i][j];
                }
            }

            // Add random points to reach k centers
            let mut selected_indices = Vec::with_capacity(k - centers.len());
            while selected_indices.len() < k - centers.len() {
                let idx = rng.random_range(0..n_samples);
                if !selected_indices.contains(&idx) {
                    selected_indices.push(idx);
                }
            }

            for (i, &idx) in selected_indices.iter().enumerate() {
                for j in 0..n_features {
                    centroids[[centers.len() + i, j]] = data[[idx, j]];
                }
            }

            Ok(centroids)
        }
        std::cmp::Ordering::Equal => {
            // We have exactly k centers
            let mut centroids = Array2::zeros((k, n_features));
            for i in 0..k {
                for j in 0..n_features {
                    centroids[[i, j]] = centers[i][j];
                }
            }
            Ok(centroids)
        }
    }
}

/// Run a single weighted k-means clustering iteration
#[allow(dead_code)]
fn _weighted_kmeans_single<F>(
    data: ArrayView2<F>,
    weights: ArrayView1<F>,
    initcentroids: ArrayView2<F>,
    opts: &KMeansOptions<F>,
) -> Result<(Array2<F>, Array1<usize>)>
where
    F: Float + FromPrimitive + Debug + std::iter::Sum,
{
    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];
    let k = initcentroids.shape()[0];

    let mut centroids = initcentroids.to_owned();
    let mut labels = Array1::zeros(n_samples);
    let mut prev_centroid_diff = F::infinity();

    for _iter in 0..opts.max_iter {
        // Assign samples to nearest centroid
        let (new_labels_, _) = vq(data, centroids.view())?;
        labels = new_labels_;

        // Compute new centroids using weights
        let mut newcentroids = Array2::zeros((k, n_features));
        let mut total_weights = Array1::zeros(k);

        for i in 0..n_samples {
            let cluster = labels[i];
            let point = data.slice(s![i, ..]);
            let weight = weights[i];

            for j in 0..n_features {
                newcentroids[[cluster, j]] = newcentroids[[cluster, j]] + point[j] * weight;
            }

            total_weights[cluster] = total_weights[cluster] + weight;
        }

        // If a cluster is empty, reinitialize it
        for i in 0..k {
            if total_weights[i] <= F::epsilon() {
                // Find the point furthest from its centroid
                let mut max_dist = F::zero();
                let mut far_idx = 0;

                for j in 0..n_samples {
                    let dist = euclidean_distance(
                        data.slice(s![j, ..]),
                        centroids.slice(s![labels[j], ..]),
                    );
                    if dist > max_dist {
                        max_dist = dist;
                        far_idx = j;
                    }
                }

                // Move this point to the empty cluster
                for j in 0..n_features {
                    newcentroids[[i, j]] = data[[far_idx, j]];
                }

                total_weights[i] = weights[far_idx];
            } else {
                // Normalize by the total weight in the cluster
                for j in 0..n_features {
                    newcentroids[[i, j]] = newcentroids[[i, j]] / total_weights[i];
                }
            }
        }

        // Check for convergence
        let mut centroid_diff = F::zero();
        for i in 0..k {
            let dist =
                euclidean_distance(centroids.slice(s![i, ..]), newcentroids.slice(s![i, ..]));
            centroid_diff = centroid_diff + dist;
        }

        centroids = newcentroids;

        if centroid_diff <= opts.tol || centroid_diff >= prev_centroid_diff {
            break;
        }

        prev_centroid_diff = centroid_diff;
    }

    Ok((centroids, labels))
}

/// Enhanced K-means clustering with custom distance metrics
///
/// This function extends the standard K-means algorithm to support various distance
/// metrics including Euclidean, Manhattan, Chebyshev, Mahalanobis, and more.
///
/// # Arguments
///
/// * `data` - Input data (n_samples × n_features)
/// * `k` - Number of clusters
/// * `metric` - Distance metric to use for clustering
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
/// use ndarray::Array2;
/// use scirs2_cluster::vq::{kmeans_with_metric, EuclideanDistance, KMeansOptions};
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
/// let metric = Box::new(EuclideanDistance);
/// let (centroids, labels) = kmeans_with_metric(data.view(), 2, metric, None).unwrap();
/// ```
#[allow(dead_code)]
pub fn kmeans_with_metric<F>(
    data: ArrayView2<F>,
    k: usize,
    metric: Box<dyn crate::vq::VQDistanceMetric<F>>,
    options: Option<KMeansOptions<F>>,
) -> Result<(Array2<F>, Array1<usize>)>
where
    F: Float + FromPrimitive + Debug + std::iter::Sum + Send + Sync + 'static,
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

    if k > n_samples {
        return Err(ClusteringError::InvalidInput(format!(
            "Number of clusters ({}) cannot be greater than number of data points ({})",
            k, n_samples
        )));
    }

    let opts = options.unwrap_or_default();

    let mut bestcentroids = None;
    let mut best_labels = None;
    let mut best_inertia = F::infinity();

    // If we're using K-means|| initialization, we only need to run once
    let n_init = if opts.init_method == KMeansInit::KMeansParallel {
        1
    } else {
        opts.n_init
    };

    for _ in 0..n_init {
        // Initialize centroids using the specified method
        let centroids = kmeans_init(data, k, Some(opts.init_method), opts.random_seed)?;

        // Run k-means with custom distance metric
        let (centroids, labels, inertia) =
            _kmeans_single_with_metric(data, centroids.view(), metric.as_ref(), &opts)?;

        if inertia < best_inertia {
            bestcentroids = Some(centroids);
            best_labels = Some(labels);
            best_inertia = inertia;
        }
    }

    Ok((bestcentroids.unwrap(), best_labels.unwrap()))
}

/// Run a single k-means clustering iteration with custom distance metric
#[allow(dead_code)]
fn _kmeans_single_with_metric<F>(
    data: ArrayView2<F>,
    initcentroids: ArrayView2<F>,
    metric: &dyn crate::vq::VQDistanceMetric<F>,
    opts: &KMeansOptions<F>,
) -> Result<(Array2<F>, Array1<usize>, F)>
where
    F: Float + FromPrimitive + Debug + std::iter::Sum + Send + Sync,
{
    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];
    let k = initcentroids.shape()[0];

    let mut centroids = initcentroids.to_owned();
    let mut labels = Array1::zeros(n_samples);
    let mut prev_centroid_diff = F::infinity();

    for _iter in 0..opts.max_iter {
        // Assign samples to nearest centroid using custom metric
        let (new_labels, distances) = _vq_with_metric(data, centroids.view(), metric)?;
        labels = new_labels;

        // Compute new centroids
        let mut newcentroids = Array2::zeros((k, n_features));
        let mut counts = Array1::zeros(k);

        for i in 0..n_samples {
            let cluster = labels[i];
            let point = data.slice(s![i, ..]);

            for j in 0..n_features {
                newcentroids[[cluster, j]] = newcentroids[[cluster, j]] + point[j];
            }

            counts[cluster] += 1;
        }

        // If a cluster is empty, reinitialize it
        for i in 0..k {
            if counts[i] == 0 {
                // Find the point furthest from its centroid
                let mut max_dist = F::zero();
                let mut far_idx = 0;

                for j in 0..n_samples {
                    let dist = distances[j];
                    if dist > max_dist {
                        max_dist = dist;
                        far_idx = j;
                    }
                }

                // Move this point to the empty cluster
                for j in 0..n_features {
                    newcentroids[[i, j]] = data[[far_idx, j]];
                }

                counts[i] = 1;
            } else {
                // Normalize by the number of points in the cluster
                for j in 0..n_features {
                    newcentroids[[i, j]] = newcentroids[[i, j]] / F::from(counts[i]).unwrap();
                }
            }
        }

        // Check for convergence using custom metric
        let mut centroid_diff = F::zero();
        for i in 0..k {
            let dist = metric.distance(centroids.slice(s![i, ..]), newcentroids.slice(s![i, ..]));
            centroid_diff = centroid_diff + dist;
        }

        centroids = newcentroids;

        if centroid_diff <= opts.tol || centroid_diff >= prev_centroid_diff {
            break;
        }

        prev_centroid_diff = centroid_diff;
    }

    // Calculate inertia (sum of squared distances to nearest centroid)
    let mut inertia = F::zero();
    for i in 0..n_samples {
        let cluster = labels[i];
        let dist = metric.distance(data.slice(s![i, ..]), centroids.slice(s![cluster, ..]));
        inertia = inertia + dist * dist;
    }

    Ok((centroids, labels, inertia))
}

/// Vector quantization with custom distance metric
#[allow(dead_code)]
fn _vq_with_metric<F>(
    data: ArrayView2<F>,
    centroids: ArrayView2<F>,
    metric: &dyn crate::vq::VQDistanceMetric<F>,
) -> Result<(Array1<usize>, Array1<F>)>
where
    F: Float + FromPrimitive + Debug + Send + Sync,
{
    let n_samples = data.shape()[0];
    let ncentroids = centroids.shape()[0];

    let mut labels = Array1::zeros(n_samples);
    let mut distances = Array1::zeros(n_samples);

    for i in 0..n_samples {
        let point = data.slice(s![i, ..]);
        let mut min_dist = F::infinity();
        let mut closest_centroid = 0;

        for j in 0..ncentroids {
            let centroid = centroids.slice(s![j, ..]);
            let dist = metric.distance(point, centroid);

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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_kmeans_random_init() {
        // Create a sample dataset
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.2, 1.8, 0.8, 1.9, 4.0, 5.0, 4.2, 4.8, 3.9, 5.1],
        )
        .unwrap();

        // Run k-means with random initialization
        let options = KMeansOptions {
            init_method: KMeansInit::Random,
            ..Default::default()
        };

        let result = kmeans_with_options(data.view(), 2, Some(options));
        assert!(result.is_ok());

        let (centroids, labels) = result.unwrap();

        // Check dimensions
        assert_eq!(centroids.shape(), &[2, 2]);
        assert_eq!(labels.len(), 6);

        // Check that we have exactly 2 clusters
        let unique_labels: std::collections::HashSet<_> = labels.iter().cloned().collect();
        assert_eq!(unique_labels.len(), 2);
    }

    #[test]
    fn test_kmeans_plusplus_init() {
        // Create a sample dataset
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.2, 1.8, 0.8, 1.9, 4.0, 5.0, 4.2, 4.8, 3.9, 5.1],
        )
        .unwrap();

        // Run k-means with k-means++ initialization
        let options = KMeansOptions {
            init_method: KMeansInit::KMeansPlusPlus,
            ..Default::default()
        };

        let result = kmeans_with_options(data.view(), 2, Some(options));
        assert!(result.is_ok());

        let (centroids, labels) = result.unwrap();

        // Check dimensions
        assert_eq!(centroids.shape(), &[2, 2]);
        assert_eq!(labels.len(), 6);

        // Check that we have exactly 2 clusters
        let unique_labels: std::collections::HashSet<_> = labels.iter().cloned().collect();
        assert_eq!(unique_labels.len(), 2);
    }

    #[test]
    fn test_kmeans_parallel_init() {
        // Create a sample dataset
        let data = Array2::from_shape_vec(
            (20, 2),
            vec![
                1.0, 2.0, 1.2, 1.8, 0.8, 1.9, 1.1, 2.2, 0.9, 1.7, 1.3, 2.1, 1.0, 1.9, 0.7, 2.0,
                1.2, 2.3, 1.5, 1.8, 5.0, 6.0, 5.2, 5.8, 4.8, 6.2, 5.1, 5.9, 5.3, 6.1, 4.9, 5.7,
                5.0, 6.3, 5.4, 5.6, 4.7, 5.9, 5.2, 6.2,
            ],
        )
        .unwrap();

        // Run k-means with k-means|| initialization
        let options = KMeansOptions {
            init_method: KMeansInit::KMeansParallel,
            ..Default::default()
        };

        let result = kmeans_with_options(data.view(), 2, Some(options));
        assert!(result.is_ok());

        let (centroids, labels) = result.unwrap();

        // Check dimensions
        assert_eq!(centroids.shape(), &[2, 2]);
        assert_eq!(labels.len(), 20);

        // Check that we have exactly 2 clusters
        let unique_labels: std::collections::HashSet<_> = labels.iter().cloned().collect();
        assert_eq!(unique_labels.len(), 2);

        // Check that the clusters are sensible (first 10 points should be in one cluster, last 10 in another)
        let first_cluster = labels[0];
        for i in 0..10 {
            assert_eq!(labels[i], first_cluster);
        }

        let second_cluster = labels[10];
        assert_ne!(first_cluster, second_cluster);
        for i in 10..20 {
            assert_eq!(labels[i], second_cluster);
        }
    }

    #[test]
    fn test_scipy_compatible_kmeans() {
        // Test the new SciPy-compatible kmeans function
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.2, 1.8, 0.8, 1.9, 4.0, 5.0, 4.2, 4.8, 3.9, 5.1],
        )
        .unwrap();

        // Test with all parameters
        let result = kmeans(
            data.view(),
            2,          // k_or_guess
            Some(20),   // iter
            Some(1e-5), // thresh
            Some(true), // check_finite
            Some(42),   // seed
        );
        assert!(result.is_ok());

        let (centroids, distortion) = result.unwrap();

        // Check dimensions
        assert_eq!(centroids.shape(), &[2, 2]);

        // Distortion should be positive
        assert!(distortion > 0.0);

        // Test with default parameters (None values)
        let result = kmeans(
            data.view(),
            2,    // k_or_guess
            None, // iter (default: 20)
            None, // thresh (default: 1e-5)
            None, // check_finite (default: true)
            None, // seed (random)
        );
        assert!(result.is_ok());

        let (centroids2, distortion2) = result.unwrap();
        assert_eq!(centroids2.shape(), &[2, 2]);
        assert!(distortion2 > 0.0);
    }

    #[test]
    fn test_scipy_kmeans_check_finite() {
        let data =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 1.5, 1.5, 8.0, 8.0, 8.5, 8.5]).unwrap();

        // Test with check_finite = true (should work with finite data)
        let result = kmeans(
            data.view(),
            2,
            Some(10),
            Some(1e-5),
            Some(true), // check_finite = true
            Some(42),
        );
        assert!(result.is_ok());

        // Test with check_finite = false (should also work with finite data)
        let result = kmeans(
            data.view(),
            2,
            Some(10),
            Some(1e-5),
            Some(false), // check_finite = false
            Some(42),
        );
        assert!(result.is_ok());
    }
}
