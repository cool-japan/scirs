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
mod kmeans2;
mod minibatch_kmeans;
pub use kmeans::*;
pub use kmeans2::{kmeans2, MinitMethod, MissingMethod};
pub use minibatch_kmeans::*;

/// Computes the Euclidean distance between two vectors
pub fn euclidean_distance<F>(x: ArrayView1<F>, y: ArrayView1<F>) -> F
where
    F: Float + FromPrimitive,
{
    let mut sum = F::zero();
    for i in 0..x.len() {
        let diff = x[i] - y[i];
        sum = sum + diff * diff;
    }
    sum.sqrt()
}

/// Normalize a group of observations on a per feature basis.
///
/// Before running k-means, it is beneficial to rescale each feature
/// dimension of the observation set by its standard deviation (i.e. "whiten"
/// it - as in "white noise" where each frequency has equal power).
/// Each feature is divided by its standard deviation across all observations
/// to give it unit variance.
///
/// # Arguments
///
/// * `obs` - Input data (n_samples × n_features)
/// * `check_finite` - Whether to check that the input contains only finite values
///
/// # Returns
///
/// * Whitened array with the same shape as input
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use scirs2_cluster::vq::whiten;
///
/// let data = Array2::<f64>::from_shape_vec((4, 2), vec![
///     1.0, 2.0,
///     1.5, 2.5,
///     0.5, 1.5,
///     2.0, 3.0,
/// ]).unwrap();
///
/// let whitened = whiten(&data).unwrap();
/// ```
pub fn whiten<F>(obs: &Array2<F>) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + std::fmt::Debug,
{
    let n_samples = obs.shape()[0];
    let n_features = obs.shape()[1];

    // Calculate mean for each feature
    let mut means = Array1::<F>::zeros(n_features);
    for j in 0..n_features {
        let mut sum = F::zero();
        for i in 0..n_samples {
            sum = sum + obs[[i, j]];
        }
        means[j] = sum / F::from(n_samples).unwrap();
    }

    // Calculate standard deviation for each feature
    let mut stds = Array1::<F>::zeros(n_features);
    for j in 0..n_features {
        let mut sum = F::zero();
        for i in 0..n_samples {
            let diff = obs[[i, j]] - means[j];
            sum = sum + diff * diff;
        }
        stds[j] = (sum / F::from(n_samples - 1).unwrap()).sqrt();

        // Avoid division by zero
        if stds[j] < F::from(1e-10).unwrap() {
            stds[j] = F::one();
        }
    }

    // Whiten the data (subtract mean and divide by std)
    let mut whitened = Array2::<F>::zeros((n_samples, n_features));
    for i in 0..n_samples {
        for j in 0..n_features {
            whitened[[i, j]] = (obs[[i, j]] - means[j]) / stds[j];
        }
    }

    Ok(whitened)
}

/// Assign codes from a code book to observations.
///
/// Assigns a code from a code book to each observation. Each
/// observation vector in the 'M' by 'N' `obs` array is compared with the
/// centroids in the code book and assigned the code of the closest
/// centroid.
///
/// # Arguments
///
/// * `data` - Input data (n_samples × n_features). Each row is an observation.
/// * `centroids` - The code book (n_centroids x n_features). Each row is a centroid.
///
/// # Returns
///
/// * Tuple of (labels, distances) where:
///   - labels: Array of shape (n_samples,) with cluster assignments
///   - distances: Array of shape (n_samples,) with distances to the closest centroid
///
/// # Errors
///
/// * Returns an error if the dimensions of data and centroids don't match
pub fn vq<F>(data: ArrayView2<F>, centroids: ArrayView2<F>) -> Result<(Array1<usize>, Array1<F>)>
where
    F: Float + FromPrimitive + Debug,
{
    if data.shape()[1] != centroids.shape()[1] {
        return Err(ClusteringError::InvalidInput(format!(
            "Observation array and centroid array must have the same number of dimensions. Got {} and {}",
            data.shape()[1], centroids.shape()[1]
        )));
    }

    let n_samples = data.shape()[0];
    let n_centroids = centroids.shape()[0];

    let mut labels = Array1::zeros(n_samples);
    let mut distances = Array1::zeros(n_samples);

    for i in 0..n_samples {
        let point = data.slice(s![i, ..]);
        let mut min_dist = F::infinity();
        let mut closest_centroid = 0;

        for j in 0..n_centroids {
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
