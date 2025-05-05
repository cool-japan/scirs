//! Vector quantization functions
//!
//! This module provides vector quantization algorithms like K-means clustering
//! and related utilities.
//!
//! ## Examples
//!
//! ```
//! use ndarray::{Array2, ArrayView2};
//! use scirs2_cluster::vq::{kmeans, KMeansInit, KMeansOptions};
//!
//! // Example data
//! let data = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 2.0,
//!     1.2, 1.8,
//!     0.8, 1.9,
//!     3.7, 4.2,
//!     3.9, 3.9,
//!     4.2, 4.1,
//! ]).unwrap();
//!
//! // Run k-means with k=2 and parallel initialization
//! let options = KMeansOptions {
//!     init_method: KMeansInit::KMeansParallel,
//!     ..Default::default()
//! };
//! let (centroids, labels) = kmeans(ArrayView2::from(&data), 2, Some(options)).unwrap();
//!
//! // Print the results
//! println!("Centroids: {:?}", centroids);
//! println!("Labels: {:?}", labels);
//! ```

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};

mod kmeans;
mod minibatch_kmeans;
pub use kmeans::*;
pub use minibatch_kmeans::*;

/// Computes the Euclidean distance between two vectors
pub fn euclidean_distance<F>(x: ArrayView1<F>, y: ArrayView1<F>) -> F
where
    F: Float + FromPrimitive,
{
    let mut sum = F::zero();
    for (xi, yi) in x.iter().zip(y.iter()) {
        let diff = *xi - *yi;
        sum = sum + diff * diff;
    }
    sum.sqrt()
}

/// Computes distances between all pairs of vectors in two sets
pub fn pairwise_distances<F>(x: ArrayView2<F>, y: ArrayView2<F>) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug,
{
    if x.shape()[1] != y.shape()[1] {
        return Err(ClusteringError::InvalidInput(
            "Vectors must have the same dimensions".to_string(),
        ));
    }

    let n_x = x.shape()[0];
    let n_y = y.shape()[0];
    let mut distances = Array2::zeros((n_x, n_y));

    for i in 0..n_x {
        for j in 0..n_y {
            distances[[i, j]] = euclidean_distance(x.slice(s![i, ..]), y.slice(s![j, ..]));
        }
    }

    Ok(distances)
}

/// Assigns data points to the nearest centroids
pub fn vq<F>(data: ArrayView2<F>, centroids: ArrayView2<F>) -> Result<(Array1<usize>, Array1<F>)>
where
    F: Float + FromPrimitive + Debug,
{
    if data.shape()[1] != centroids.shape()[1] {
        return Err(ClusteringError::InvalidInput(
            "Data and centroids must have the same dimensions".to_string(),
        ));
    }

    let n_samples = data.shape()[0];
    let n_clusters = centroids.shape()[0];

    let mut labels = Array1::zeros(n_samples);
    let mut distances = Array1::zeros(n_samples);

    for i in 0..n_samples {
        let point = data.slice(s![i, ..]);
        let mut min_dist = F::infinity();
        let mut min_idx = 0;

        for j in 0..n_clusters {
            let centroid = centroids.slice(s![j, ..]);
            let dist = euclidean_distance(point, centroid);

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
