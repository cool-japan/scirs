//! Density-based clustering algorithms
//!
//! This module provides implementations of density-based clustering algorithms such as:
//! - DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
//! - OPTICS (Ordering Points To Identify the Clustering Structure)
//! - HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)
//!
//! These algorithms are particularly useful for discovering clusters of arbitrary shape
//! and for identifying noise points in the data. Density-based methods are also capable
//! of finding clusters of arbitrary shapes and handling clusters of different densities.

pub mod hdbscan;
pub mod optics;

use ndarray::{Array1, Array2, ArrayView2};
use num_traits::{Float, FromPrimitive};
// Directly implement distance functions
mod distance {
    use num_traits::Float;

    /// Calculates the Euclidean distance (L2 norm) between two vectors
    pub fn euclidean<F: Float>(a: &[F], b: &[F]) -> F {
        let mut sum = F::zero();
        for (a_i, b_i) in a.iter().zip(b.iter()) {
            let diff = *a_i - *b_i;
            sum = sum + diff * diff;
        }
        sum.sqrt()
    }

    /// Calculates the Manhattan distance (L1 norm) between two vectors
    pub fn manhattan<F: Float>(a: &[F], b: &[F]) -> F {
        let mut sum = F::zero();
        for (a_i, b_i) in a.iter().zip(b.iter()) {
            sum = sum + (*a_i - *b_i).abs();
        }
        sum
    }

    /// Calculates the Chebyshev distance (L∞ norm) between two vectors
    pub fn chebyshev<F: Float>(a: &[F], b: &[F]) -> F {
        let mut max = F::zero();
        for (a_i, b_i) in a.iter().zip(b.iter()) {
            let abs_diff = (*a_i - *b_i).abs();
            if abs_diff > max {
                max = abs_diff;
            }
        }
        max
    }

    /// Calculates the Minkowski distance between two vectors with power p
    pub fn minkowski<F: Float>(a: &[F], b: &[F], p: F) -> F {
        let mut sum = F::zero();
        for (a_i, b_i) in a.iter().zip(b.iter()) {
            sum = sum + ((*a_i - *b_i).abs()).powf(p);
        }
        sum.powf(F::one() / p)
    }
}
use std::collections::HashSet;
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};

/// Labels for DBSCAN clusters
pub mod labels {
    /// Noise point label (-1)
    pub const NOISE: i32 = -1;
    /// Undefined point label (not yet processed)
    pub const UNDEFINED: i32 = -2;
}

/// Distance metric enumeration for clustering algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Euclidean distance (L2 norm)
    Euclidean,

    /// Manhattan distance (L1 norm)
    Manhattan,

    /// Chebyshev distance (L∞ norm)
    Chebyshev,

    /// Minkowski distance with p=3
    Minkowski,
}

/// DBSCAN clustering algorithm (Density-Based Spatial Clustering of Applications with Noise)
///
/// # Arguments
///
/// * `data` - The input data as a 2D array (n_samples x n_features)
/// * `eps` - The maximum distance between two samples for them to be considered neighbors
/// * `min_samples` - The minimum number of samples in a neighborhood for a point to be a core point
/// * `metric` - The distance metric to use (default: Euclidean)
///
/// # Returns
///
/// * `Result<Array1<i32>>` - Cluster labels for each point in the dataset.
///   Labels are integers starting from 0, with -1 indicating noise points.
///
/// # Examples
///
/// ```
/// use ndarray::{Array2, ArrayView2};
/// use scirs2_cluster::density::{dbscan, DistanceMetric};
///
/// // Example data with two clusters and noise
/// let data = Array2::from_shape_vec((8, 2), vec![
///     1.0, 2.0,   // Cluster 1
///     1.5, 1.8,   // Cluster 1
///     1.3, 1.9,   // Cluster 1
///     5.0, 7.0,   // Cluster 2
///     5.1, 6.8,   // Cluster 2
///     5.2, 7.1,   // Cluster 2
///     0.0, 10.0,  // Noise
///     10.0, 0.0,  // Noise
/// ]).unwrap();
///
/// // Run DBSCAN with eps=0.8 and min_samples=2
/// let labels = dbscan(data.view(), 0.8, 2, Some(DistanceMetric::Euclidean)).unwrap();
///
/// // Print the results
/// println!("Cluster assignments: {:?}", labels);
/// ```
pub fn dbscan<F: Float + FromPrimitive + Debug + PartialOrd>(
    data: ArrayView2<F>,
    eps: F,
    min_samples: usize,
    metric: Option<DistanceMetric>,
) -> Result<Array1<i32>> {
    let n_samples = data.shape()[0];

    if n_samples == 0 {
        return Err(ClusteringError::InvalidInput("Empty input data".into()));
    }

    if eps <= F::zero() {
        return Err(ClusteringError::InvalidInput("eps must be positive".into()));
    }

    if min_samples < 1 {
        return Err(ClusteringError::InvalidInput(
            "min_samples must be at least 1".into(),
        ));
    }

    // Initialize labels to undefined
    let mut labels = vec![labels::UNDEFINED; n_samples];

    // Keep track of the current cluster label
    let mut cluster_label: i32 = 0;

    // Calculate pairwise distances
    let mut distances = Array2::<F>::zeros((n_samples, n_samples));

    // Fill distance matrix
    for i in 0..n_samples {
        for j in (i + 1)..n_samples {
            let point1 = data.row(i).to_vec();
            let point2 = data.row(j).to_vec();

            let dist = match metric.unwrap_or(DistanceMetric::Euclidean) {
                DistanceMetric::Euclidean => distance::euclidean(&point1, &point2),
                DistanceMetric::Manhattan => distance::manhattan(&point1, &point2),
                DistanceMetric::Chebyshev => distance::chebyshev(&point1, &point2),
                DistanceMetric::Minkowski => {
                    distance::minkowski(&point1, &point2, F::from(3.0).unwrap())
                }
            };

            distances[[i, j]] = dist;
            distances[[j, i]] = dist; // Distance matrix is symmetric
        }
    }

    // Create a map of neighbor indices for each point
    let mut neighborhoods: Vec<Vec<usize>> = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let mut neighbors = Vec::new();
        for j in 0..n_samples {
            if i != j && distances[[i, j]] <= eps {
                neighbors.push(j);
            }
        }
        neighborhoods.push(neighbors);
    }

    // Main DBSCAN algorithm
    for point_idx in 0..n_samples {
        // Skip already processed points
        if labels[point_idx] != labels::UNDEFINED {
            continue;
        }

        // Check if point has enough neighbors to be a core point
        let neighbors = &neighborhoods[point_idx];

        if neighbors.len() + 1 < min_samples {
            // Mark as noise (may be assigned to a cluster later as a border point)
            labels[point_idx] = labels::NOISE;
            continue;
        }

        // Start a new cluster
        labels[point_idx] = cluster_label;

        // Process neighbors using a queue (breadth-first search)
        let mut queue: Vec<usize> = neighbors.clone();
        let mut processed = HashSet::new();
        processed.insert(point_idx);

        let mut idx = 0;
        while idx < queue.len() {
            let current = queue[idx];
            idx += 1;

            // Skip already processed points
            if processed.contains(&current) {
                continue;
            }
            processed.insert(current);

            // If was previously noise, add to cluster
            if labels[current] == labels::NOISE {
                labels[current] = cluster_label;
                continue;
            }

            // If already part of another cluster, skip
            if labels[current] != labels::UNDEFINED {
                continue;
            }

            // Add to current cluster
            labels[current] = cluster_label;

            // If it's a core point, add its neighbors to the queue
            let current_neighbors = &neighborhoods[current];
            if current_neighbors.len() + 1 >= min_samples {
                for &neighbor in current_neighbors {
                    if !processed.contains(&neighbor) {
                        queue.push(neighbor);
                    }
                }
            }
        }

        // Move to next cluster
        cluster_label += 1;
    }

    Ok(Array1::from(labels))
}

/// Runs the OPTICS clustering algorithm
///
/// This is a wrapper around the more detailed implementation in the optics module.
/// See `optics::optics` for more details.
///
/// # Arguments
///
/// * `data` - The input data as a 2D array (n_samples x n_features)
/// * `min_samples` - The minimum number of samples required for a point to be a core point
/// * `max_eps` - The maximum distance for neighborhood consideration (default: infinity)
/// * `metric` - Distance metric to use (default: Euclidean)
///
/// # Returns
///
/// * `Result<Array1<i32>>` - Cluster labels from a default extraction (using DBSCAN-like extraction
///   with a reasonable epsilon value), or an error if the computation fails.
pub fn optics<F: Float + FromPrimitive + Debug + PartialOrd>(
    data: ArrayView2<F>,
    min_samples: usize,
    max_eps: Option<F>,
    metric: Option<DistanceMetric>,
) -> Result<Array1<i32>> {
    let result = optics::optics(data, min_samples, max_eps, metric)?;

    // Convert max_eps to f64 for use with default extraction
    let default_eps = match max_eps {
        Some(eps) => eps.to_f64().unwrap() / 10.0, // Use smaller epsilon by default
        None => 0.5,                               // Default eps if none provided
    };

    Ok(optics::extract_dbscan_clustering(&result, default_eps))
}
