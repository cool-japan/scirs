//! Enhanced K-means clustering implementation with multiple initialization methods

use ndarray::{s, Array1, Array2, ArrayView2};
use num_traits::{Float, FromPrimitive};
use rand::{rng, rngs::StdRng, Rng, RngCore, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::fmt::Debug;

use super::{euclidean_distance, vq};
use crate::error::{ClusteringError, Result};

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
    F: Float + FromPrimitive + Debug + std::iter::Sum,
{
    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];
    let iterations = iter.unwrap_or(10);
    let _threshold = thresh.unwrap_or(F::from(1e-5).unwrap());
    let missing_method = missing.unwrap_or(MissingMethod::Warn);
    let _check_finite = check_finite.unwrap_or(true);

    if k == 0 || k > n_samples {
        return Err(ClusteringError::InvalidInput(format!(
            "Number of clusters ({}) must be between 1 and number of samples ({})",
            k, n_samples
        )));
    }

    // Initialize centroids
    let mut centroids = if let Some(init_method) = minit {
        match init_method {
            MinitMethod::Random => krandinit(data, k, random_seed)?,
            MinitMethod::Points => kpoints(data, k, random_seed)?,
            MinitMethod::PlusPlus => kmeans_plus_plus(data, k, random_seed)?,
        }
    } else {
        // Handle case where k is provided as initial centroids array
        // This is handled by the caller and should not reach here for now
        return Err(ClusteringError::InvalidInput(
            "Direct centroid initialization not yet supported".to_string(),
        ));
    };

    let mut labels;

    // Run K-means iterations
    for _iteration in 0..iterations {
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
    }

    // Final assignment
    let (final_labels, _) = vq(data, centroids.view())?;

    Ok((centroids, final_labels))
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
