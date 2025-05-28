//! Parallel K-means clustering implementation using Rayon
//!
//! This module provides a parallel implementation of the K-means clustering algorithm
//! that leverages multiple CPU cores for improved performance on large datasets.

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use num_traits::{Float, FromPrimitive};
use rayon::prelude::*;
use std::fmt::Debug;
use std::sync::Mutex;

use super::{euclidean_distance, kmeans_init, KMeansInit};
use crate::error::{ClusteringError, Result};

/// Options for parallel K-means clustering
#[derive(Debug, Clone)]
pub struct ParallelKMeansOptions<F: Float> {
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
    /// Number of threads to use (None = use all available)
    pub n_threads: Option<usize>,
}

impl<F: Float + FromPrimitive> Default for ParallelKMeansOptions<F> {
    fn default() -> Self {
        Self {
            max_iter: 300,
            tol: F::from(1e-4).unwrap(),
            random_seed: None,
            n_init: 10,
            init_method: KMeansInit::KMeansPlusPlus,
            n_threads: None,
        }
    }
}

/// Parallel K-means clustering algorithm
///
/// This implementation parallelizes the assignment step and centroid computation
/// for improved performance on multi-core systems.
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
/// use ndarray::Array2;
/// use scirs2_cluster::vq::parallel_kmeans;
///
/// let data = Array2::from_shape_vec((1000, 2),
///     (0..2000).map(|i| i as f64 / 100.0).collect()
/// ).unwrap();
///
/// let (centroids, labels) = parallel_kmeans(data.view(), 5, None).unwrap();
/// ```
pub fn parallel_kmeans<F>(
    data: ArrayView2<F>,
    k: usize,
    options: Option<ParallelKMeansOptions<F>>,
) -> Result<(Array2<F>, Array1<usize>)>
where
    F: Float + FromPrimitive + Debug + std::iter::Sum + Send + Sync,
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

    // Set thread pool size if specified
    if let Some(n_threads) = opts.n_threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build_global()
            .ok();
    }

    let mut best_centroids = None;
    let mut best_labels = None;
    let mut best_inertia = F::infinity();

    // Run multiple initializations
    for _ in 0..opts.n_init {
        // Initialize centroids
        let centroids = kmeans_init(data, k, Some(opts.init_method), opts.random_seed)?;

        // Run parallel k-means
        let (centroids, labels, inertia) = parallel_kmeans_single(data, centroids.view(), &opts)?;

        if inertia < best_inertia {
            best_centroids = Some(centroids);
            best_labels = Some(labels);
            best_inertia = inertia;
        }
    }

    Ok((best_centroids.unwrap(), best_labels.unwrap()))
}

/// Run a single parallel k-means clustering iteration
fn parallel_kmeans_single<F>(
    data: ArrayView2<F>,
    init_centroids: ArrayView2<F>,
    opts: &ParallelKMeansOptions<F>,
) -> Result<(Array2<F>, Array1<usize>, F)>
where
    F: Float + FromPrimitive + Debug + std::iter::Sum + Send + Sync,
{
    let n_samples = data.shape()[0];
    let _n_features = data.shape()[1];
    let k = init_centroids.shape()[0];

    let mut centroids = init_centroids.to_owned();
    let mut labels = Array1::zeros(n_samples);
    let mut prev_inertia = F::infinity();

    for _iter in 0..opts.max_iter {
        // Parallel assignment step
        let (new_labels, distances) = parallel_assign_labels(data, centroids.view())?;
        labels = new_labels;

        // Parallel centroid update
        let new_centroids = parallel_update_centroids(data, &labels, k)?;

        // Check for empty clusters
        let cluster_counts = count_clusters(&labels, k);

        // Handle empty clusters
        let mut final_centroids = new_centroids;
        for (i, &count) in cluster_counts.iter().enumerate() {
            if count == 0 {
                // Find the point furthest from its centroid
                let (far_idx, _) = distances
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap();

                // Move this point to the empty cluster
                final_centroids
                    .slice_mut(s![i, ..])
                    .assign(&data.slice(s![far_idx, ..]));
            }
        }

        // Calculate inertia
        let inertia = parallel_compute_inertia(data, &labels, final_centroids.view())?;

        // Check for convergence
        if (prev_inertia - inertia).abs() <= opts.tol {
            return Ok((final_centroids, labels, inertia));
        }

        centroids = final_centroids;
        prev_inertia = inertia;
    }

    // Final inertia calculation
    let final_inertia = parallel_compute_inertia(data, &labels, centroids.view())?;

    Ok((centroids, labels, final_inertia))
}

/// Parallel assignment of samples to nearest centroids
fn parallel_assign_labels<F>(
    data: ArrayView2<F>,
    centroids: ArrayView2<F>,
) -> Result<(Array1<usize>, Array1<F>)>
where
    F: Float + FromPrimitive + Debug + Send + Sync,
{
    let n_samples = data.shape()[0];
    let k = centroids.shape()[0];

    // Parallel computation of labels and distances
    let results: Vec<(usize, F)> = (0..n_samples)
        .into_par_iter()
        .map(|i| {
            let sample = data.slice(s![i, ..]);
            let mut min_dist = F::infinity();
            let mut best_label = 0;

            for j in 0..k {
                let centroid = centroids.slice(s![j, ..]);
                let dist = euclidean_distance(sample, centroid);

                if dist < min_dist {
                    min_dist = dist;
                    best_label = j;
                }
            }

            (best_label, min_dist)
        })
        .collect();

    // Convert to arrays
    let mut labels = Array1::zeros(n_samples);
    let mut distances = Array1::zeros(n_samples);

    for (i, (label, dist)) in results.into_iter().enumerate() {
        labels[i] = label;
        distances[i] = dist;
    }

    Ok((labels, distances))
}

/// Parallel update of centroids based on assigned labels
fn parallel_update_centroids<F>(
    data: ArrayView2<F>,
    labels: &Array1<usize>,
    k: usize,
) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Send + Sync + std::iter::Sum,
{
    let n_features = data.shape()[1];

    // Create thread-safe accumulators for each cluster
    let sums: Vec<Mutex<Array1<F>>> = (0..k)
        .map(|_| Mutex::new(Array1::zeros(n_features)))
        .collect();

    let counts: Vec<Mutex<usize>> = (0..k).map(|_| Mutex::new(0)).collect();

    // Parallel accumulation
    data.axis_iter(Axis(0))
        .zip(labels.iter())
        .par_bridge()
        .for_each(|(sample, &label)| {
            let mut sum = sums[label].lock().unwrap();
            for i in 0..n_features {
                sum[i] = sum[i] + sample[i];
            }

            let mut count = counts[label].lock().unwrap();
            *count += 1;
        });

    // Compute new centroids
    let mut new_centroids = Array2::zeros((k, n_features));

    for i in 0..k {
        let sum = sums[i].lock().unwrap();
        let count = *counts[i].lock().unwrap();

        if count > 0 {
            for j in 0..n_features {
                new_centroids[[i, j]] = sum[j] / F::from(count).unwrap();
            }
        }
    }

    Ok(new_centroids)
}

/// Count number of points in each cluster
fn count_clusters(labels: &Array1<usize>, k: usize) -> Vec<usize> {
    let mut counts = vec![0; k];
    for &label in labels.iter() {
        counts[label] += 1;
    }
    counts
}

/// Parallel computation of inertia (sum of squared distances to centroids)
fn parallel_compute_inertia<F>(
    data: ArrayView2<F>,
    labels: &Array1<usize>,
    centroids: ArrayView2<F>,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug + Send + Sync + std::iter::Sum,
{
    let inertia: F = data
        .axis_iter(Axis(0))
        .zip(labels.iter())
        .par_bridge()
        .map(|(sample, &label)| {
            let centroid = centroids.slice(s![label, ..]);
            let dist = euclidean_distance(sample.view(), centroid);
            dist * dist
        })
        .sum();

    Ok(inertia)
}

/// Compute squared Euclidean distance between two vectors (for performance)
#[inline]
#[allow(dead_code)]
fn squared_euclidean_distance<F>(x: ArrayView1<F>, y: ArrayView1<F>) -> F
where
    F: Float + FromPrimitive,
{
    let mut sum = F::zero();
    for i in 0..x.len() {
        let diff = x[i] - y[i];
        sum = sum + diff * diff;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_parallel_kmeans_simple() {
        // Create a simple dataset
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.2, 1.8, 0.8, 1.9, 4.0, 5.0, 4.2, 4.8, 3.9, 5.1],
        )
        .unwrap();

        // Run parallel k-means
        let options = ParallelKMeansOptions {
            n_init: 1,
            random_seed: Some(42),
            ..Default::default()
        };

        let (centroids, labels) = parallel_kmeans(data.view(), 2, Some(options)).unwrap();

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
    fn test_parallel_kmeans_large_dataset() {
        // Create a larger dataset
        let n_samples = 1000;
        let n_features = 10;

        let mut data_vec = Vec::with_capacity(n_samples * n_features);
        for i in 0..n_samples {
            for j in 0..n_features {
                // Create 3 clusters
                let cluster = i / (n_samples / 3);
                let value = (cluster * 10) as f64 + (j as f64 + i as f64 * 0.01);
                data_vec.push(value);
            }
        }

        let data = Array2::from_shape_vec((n_samples, n_features), data_vec).unwrap();

        // Run parallel k-means
        let options = ParallelKMeansOptions {
            n_init: 3,
            max_iter: 50,
            random_seed: Some(42),
            ..Default::default()
        };

        let start_time = std::time::Instant::now();
        let (centroids, labels) = parallel_kmeans(data.view(), 3, Some(options)).unwrap();
        let duration = start_time.elapsed();

        println!("Parallel K-means took: {:?}", duration);

        // Check results
        assert_eq!(centroids.shape(), &[3, n_features]);
        assert_eq!(labels.len(), n_samples);

        // Verify all samples are assigned
        for &label in labels.iter() {
            assert!(label < 3);
        }
    }
}
