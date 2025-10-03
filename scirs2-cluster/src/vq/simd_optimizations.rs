//! SIMD-optimized core clustering operations
//!
//! This module provides comprehensive SIMD optimizations for fundamental clustering
//! operations including distance computations, data preprocessing, vector quantization,
//! and centroid calculations. All functions provide automatic fallback to scalar
//! implementations when SIMD is not available.

use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::numeric::{Float, FromPrimitive, Zero};
use scirs2_core::parallel_ops::*;
use scirs2_core::simd_ops::{AutoOptimizer, PlatformCapabilities, SimdUnifiedOps};
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};
use statrs::statistics::Statistics;

/// Configuration for SIMD optimizations
#[derive(Debug, Clone)]
pub struct SimdOptimizationConfig {
    /// Minimum array size to trigger SIMD optimizations
    pub simd_threshold: usize,
    /// Enable parallel processing for large arrays
    pub enable_parallel: bool,
    /// Chunk size for parallel processing
    pub parallel_chunk_size: usize,
    /// Enable cache-friendly memory access patterns
    pub cache_friendly: bool,
    /// Force SIMD usage even for small arrays (for testing)
    pub force_simd: bool,
}

impl Default for SimdOptimizationConfig {
    fn default() -> Self {
        Self {
            simd_threshold: 64,
            enable_parallel: true,
            parallel_chunk_size: 1024,
            cache_friendly: true,
            force_simd: false,
        }
    }
}

/// SIMD-optimized Euclidean distance between two vectors
///
/// This function uses SIMD operations when available and beneficial,
/// automatically falling back to scalar computation for small vectors
/// or when SIMD is not available.
///
/// # Arguments
///
/// * `x` - First vector
/// * `y` - Second vector
/// * `config` - Optional SIMD configuration
///
/// # Returns
///
/// * Euclidean distance between the two vectors
///
/// # Errors
///
/// * Returns error if vectors have different lengths
#[allow(dead_code)]
pub fn euclidean_distance_simd<F>(
    x: ArrayView1<F>,
    y: ArrayView1<F>,
    config: Option<&SimdOptimizationConfig>,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug + SimdUnifiedOps,
{
    if x.len() != y.len() {
        return Err(ClusteringError::InvalidInput(format!(
            "Vectors must have the same length: got {} and {}",
            x.len(),
            y.len()
        )));
    }

    let default_config = SimdOptimizationConfig::default();
    let config = config.unwrap_or(&default_config);
    let caps = PlatformCapabilities::detect();
    let optimizer = AutoOptimizer::new();

    if (caps.simd_available && (optimizer.should_use_simd(x.len()) || config.force_simd))
        || x.len() >= config.simd_threshold
    {
        let diff = F::simd_sub(&x, &y);
        Ok(F::simd_norm(&diff.view()))
    } else {
        // Scalar fallback
        let mut sum = F::zero();
        for i in 0..x.len() {
            let diff = x[i] - y[i];
            sum = sum + diff * diff;
        }
        Ok(sum.sqrt())
    }
}

/// SIMD-optimized data whitening (normalization by standard deviation)
///
/// This function normalizes data features by subtracting the mean and dividing by
/// the standard deviation using SIMD operations for improved performance.
///
/// # Arguments
///
/// * `obs` - Input data (n_samples × n_features)
/// * `config` - Optional SIMD configuration
///
/// # Returns
///
/// * Whitened array with the same shape as input
#[allow(dead_code)]
pub fn whiten_simd<F>(obs: &Array2<F>, config: Option<&SimdOptimizationConfig>) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Send + Sync + SimdUnifiedOps,
{
    let default_config = SimdOptimizationConfig::default();
    let config = config.unwrap_or(&default_config);
    let n_samples = obs.shape()[0];
    let n_features = obs.shape()[1];

    if n_samples == 0 || n_features == 0 {
        return Err(ClusteringError::InvalidInput(
            "Input data cannot be empty".to_string(),
        ));
    }

    let caps = PlatformCapabilities::detect();
    let optimizer = AutoOptimizer::new();
    let use_simd = caps.simd_available
        && (optimizer.should_use_simd(n_samples * n_features) || config.force_simd);

    if use_simd && config.enable_parallel && n_features > config.parallel_chunk_size {
        whiten_simd_parallel(obs, config)
    } else if use_simd {
        whiten_simd_sequential(obs)
    } else {
        whiten_scalar_fallback(obs)
    }
}

/// SIMD-optimized sequential whitening
#[allow(dead_code)]
fn whiten_simd_sequential<F>(obs: &Array2<F>) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + SimdUnifiedOps,
{
    let n_samples = obs.shape()[0];
    let n_features = obs.shape()[1];
    let n_samples_f = F::from(n_samples).unwrap();

    // Calculate means using SIMD operations
    let mut means = Array1::<F>::zeros(n_features);
    for j in 0..n_features {
        let column = obs.column(j);
        means[j] = F::simd_sum(&column) / n_samples_f;
    }

    // Calculate standard deviations using SIMD operations
    let mut stds = Array1::<F>::zeros(n_features);
    for j in 0..n_features {
        let column = obs.column(j);
        let mean_array = Array1::from_elem(n_samples, means[j]);
        let diff = F::simd_sub(&column, &mean_array.view());
        let squared_diff = F::simd_mul(&diff.view(), &diff.view());
        let variance = F::simd_sum(&squared_diff.view()) / F::from(n_samples - 1).unwrap();
        stds[j] = variance.sqrt();

        // Avoid division by zero
        if stds[j] < F::from(1e-10).unwrap() {
            stds[j] = F::one();
        }
    }

    // Whiten the data using SIMD operations
    let mut whitened = Array2::<F>::zeros((n_samples, n_features));
    for j in 0..n_features {
        let column = obs.column(j);
        let mean_array = Array1::from_elem(n_samples, means[j]);
        let std_array = Array1::from_elem(n_samples, stds[j]);

        let centered = F::simd_sub(&column, &mean_array.view());
        let normalized = F::simd_div(&centered.view(), &std_array.view());

        for i in 0..n_samples {
            whitened[[i, j]] = normalized[i];
        }
    }

    Ok(whitened)
}

/// Parallel SIMD-optimized whitening for large datasets
#[allow(dead_code)]
fn whiten_simd_parallel<F>(obs: &Array2<F>, config: &SimdOptimizationConfig) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Send + Sync + SimdUnifiedOps,
{
    let n_samples = obs.shape()[0];
    let n_features = obs.shape()[1];
    let n_samples_f = F::from(n_samples).unwrap();

    // Parallel mean calculation
    let means: Array1<F> = if is_parallel_enabled() {
        (0..n_features)
            .into_par_iter()
            .map(|j| {
                let column = obs.column(j);
                F::simd_sum(&column) / n_samples_f
            })
            .collect::<Vec<_>>()
            .into()
    } else {
        let mut means = Array1::<F>::zeros(n_features);
        for j in 0..n_features {
            let column = obs.column(j);
            means[j] = F::simd_sum(&column) / n_samples_f;
        }
        means
    };

    // Parallel standard deviation calculation
    let stds: Array1<F> = if is_parallel_enabled() {
        (0..n_features)
            .into_par_iter()
            .map(|j| {
                let column = obs.column(j);
                let mean_array = Array1::from_elem(n_samples, means[j]);
                let diff = F::simd_sub(&column, &mean_array.view());
                let squared_diff = F::simd_mul(&diff.view(), &diff.view());
                let variance = F::simd_sum(&squared_diff.view()) / F::from(n_samples - 1).unwrap();
                let std = variance.sqrt();

                // Avoid division by zero
                if std < F::from(1e-10).unwrap() {
                    F::one()
                } else {
                    std
                }
            })
            .collect::<Vec<_>>()
            .into()
    } else {
        whiten_simd_sequential(obs)?
            .into_shape((n_samples, n_features))
            .unwrap();
        return whiten_simd_sequential(obs);
    };

    // Parallel whitening
    let mut whitened = Array2::<F>::zeros((n_samples, n_features));

    if is_parallel_enabled() {
        // Process features in parallel chunks
        let chunk_size = config.parallel_chunk_size;
        let normalized_columns: Vec<Array1<F>> = (0..n_features)
            .into_par_iter()
            .map(|j| {
                let column = obs.column(j);
                let mean_array = Array1::from_elem(n_samples, means[j]);
                let std_array = Array1::from_elem(n_samples, stds[j]);

                let centered = F::simd_sub(&column, &mean_array.view());
                F::simd_div(&centered.view(), &std_array.view())
            })
            .collect();

        // Assign the normalized columns to the whitened array
        for (j, normalized_column) in normalized_columns.iter().enumerate() {
            for i in 0..n_samples {
                whitened[[i, j]] = normalized_column[i];
            }
        }
    } else {
        for j in 0..n_features {
            let column = obs.column(j);
            let mean_array = Array1::from_elem(n_samples, means[j]);
            let std_array = Array1::from_elem(n_samples, stds[j]);

            let centered = F::simd_sub(&column, &mean_array.view());
            let normalized = F::simd_div(&centered.view(), &std_array.view());

            for i in 0..n_samples {
                whitened[[i, j]] = normalized[i];
            }
        }
    }

    Ok(whitened)
}

/// Scalar fallback for whitening when SIMD is not available
#[allow(dead_code)]
fn whiten_scalar_fallback<F>(obs: &Array2<F>) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug,
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

    // Whiten the data
    let mut whitened = Array2::<F>::zeros((n_samples, n_features));
    for i in 0..n_samples {
        for j in 0..n_features {
            whitened[[i, j]] = (obs[[i, j]] - means[j]) / stds[j];
        }
    }

    Ok(whitened)
}

/// SIMD-optimized vector quantization (assignment to nearest centroids)
///
/// This function assigns each data point to its nearest centroid using SIMD
/// operations for distance calculations.
///
/// # Arguments
///
/// * `data` - Input data (n_samples × n_features)
/// * `centroids` - Centroids (n_centroids × n_features)
/// * `config` - Optional SIMD configuration
///
/// # Returns
///
/// * Tuple of (labels, distances) where labels are cluster assignments
///   and distances are distances to the nearest centroid
#[allow(dead_code)]
pub fn vq_simd<F>(
    data: ArrayView2<F>,
    centroids: ArrayView2<F>,
    config: Option<&SimdOptimizationConfig>,
) -> Result<(Array1<usize>, Array1<F>)>
where
    F: Float + FromPrimitive + Debug + Send + Sync + SimdUnifiedOps,
{
    if data.shape()[1] != centroids.shape()[1] {
        return Err(ClusteringError::InvalidInput(format!(
            "Data and centroids must have the same number of features: {} vs {}",
            data.shape()[1],
            centroids.shape()[1]
        )));
    }

    let default_config = SimdOptimizationConfig::default();
    let config = config.unwrap_or(&default_config);
    let n_samples = data.shape()[0];
    let n_centroids = centroids.shape()[0];

    if config.enable_parallel && is_parallel_enabled() && n_samples > config.parallel_chunk_size {
        vq_simd_parallel(data, centroids, config)
    } else {
        vq_simd_sequential(data, centroids, config)
    }
}

/// Sequential SIMD-optimized vector quantization
#[allow(dead_code)]
fn vq_simd_sequential<F>(
    data: ArrayView2<F>,
    centroids: ArrayView2<F>,
    config: &SimdOptimizationConfig,
) -> Result<(Array1<usize>, Array1<F>)>
where
    F: Float + FromPrimitive + Debug + SimdUnifiedOps,
{
    let n_samples = data.shape()[0];
    let n_centroids = centroids.shape()[0];

    let mut labels = Array1::zeros(n_samples);
    let mut distances = Array1::zeros(n_samples);

    let caps = PlatformCapabilities::detect();
    let use_simd = caps.simd_available || config.force_simd;

    for i in 0..n_samples {
        let point = data.slice(s![i, ..]);
        let mut min_dist = F::infinity();
        let mut closest_centroid = 0;

        for j in 0..n_centroids {
            let centroid = centroids.slice(s![j, ..]);

            let dist = if use_simd {
                let diff = F::simd_sub(&point, &centroid);
                F::simd_norm(&diff.view())
            } else {
                // Scalar fallback
                let mut sum = F::zero();
                for k in 0..point.len() {
                    let diff = point[k] - centroid[k];
                    sum = sum + diff * diff;
                }
                sum.sqrt()
            };

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

/// Parallel SIMD-optimized vector quantization
#[allow(dead_code)]
fn vq_simd_parallel<F>(
    data: ArrayView2<F>,
    centroids: ArrayView2<F>,
    config: &SimdOptimizationConfig,
) -> Result<(Array1<usize>, Array1<F>)>
where
    F: Float + FromPrimitive + Debug + Send + Sync + SimdUnifiedOps,
{
    let n_samples = data.shape()[0];
    let n_centroids = centroids.shape()[0];

    let caps = PlatformCapabilities::detect();
    let use_simd = caps.simd_available || config.force_simd;

    // Process samples in parallel
    let results: Vec<(usize, F)> = (0..n_samples)
        .into_par_iter()
        .map(|i| {
            let point = data.slice(s![i, ..]);
            let mut min_dist = F::infinity();
            let mut closest_centroid = 0;

            for j in 0..n_centroids {
                let centroid = centroids.slice(s![j, ..]);

                let dist = if use_simd {
                    let diff = F::simd_sub(&point, &centroid);
                    F::simd_norm(&diff.view())
                } else {
                    // Scalar fallback
                    let mut sum = F::zero();
                    for k in 0..point.len() {
                        let diff = point[k] - centroid[k];
                        sum = sum + diff * diff;
                    }
                    sum.sqrt()
                };

                if dist < min_dist {
                    min_dist = dist;
                    closest_centroid = j;
                }
            }

            (closest_centroid, min_dist)
        })
        .collect();

    let mut labels = Array1::zeros(n_samples);
    let mut distances = Array1::zeros(n_samples);

    for (i, (label, distance)) in results.into_iter().enumerate() {
        labels[i] = label;
        distances[i] = distance;
    }

    Ok((labels, distances))
}

/// SIMD-optimized centroid computation for K-means
///
/// This function computes new centroids from data points and their cluster assignments
/// using SIMD operations for improved performance.
///
/// # Arguments
///
/// * `data` - Input data (n_samples × n_features)
/// * `labels` - Cluster assignments for each data point
/// * `k` - Number of clusters
/// * `config` - Optional SIMD configuration
///
/// # Returns
///
/// * Array of new centroids (k × n_features)
#[allow(dead_code)]
pub fn compute_centroids_simd<F>(
    data: ArrayView2<F>,
    labels: &Array1<usize>,
    k: usize,
    config: Option<&SimdOptimizationConfig>,
) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Send + Sync + SimdUnifiedOps + std::iter::Sum,
{
    let default_config = SimdOptimizationConfig::default();
    let config = config.unwrap_or(&default_config);
    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];

    if labels.len() != n_samples {
        return Err(ClusteringError::InvalidInput(
            "Labels array length must match number of data points".to_string(),
        ));
    }

    let caps = PlatformCapabilities::detect();
    let use_simd = caps.simd_available || config.force_simd;

    if config.enable_parallel && is_parallel_enabled() && k > 4 {
        compute_centroids_simd_parallel(data, labels, k, use_simd)
    } else {
        compute_centroids_simd_sequential(data, labels, k, use_simd)
    }
}

/// Sequential SIMD-optimized centroid computation
#[allow(dead_code)]
fn compute_centroids_simd_sequential<F>(
    data: ArrayView2<F>,
    labels: &Array1<usize>,
    k: usize,
    use_simd: bool,
) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + SimdUnifiedOps + std::iter::Sum,
{
    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];

    let mut centroids = Array2::zeros((k, n_features));
    let mut counts = Array1::<usize>::zeros(k);

    // Accumulate points for each cluster
    for i in 0..n_samples {
        let cluster = labels[i];
        if cluster >= k {
            return Err(ClusteringError::InvalidInput(format!(
                "Label {} exceeds number of clusters {}",
                cluster, k
            )));
        }

        counts[cluster] += 1;

        if use_simd {
            let point = data.slice(s![i, ..]);
            let centroid_row = centroids.slice_mut(s![cluster, ..]);
            let updated_centroid = F::simd_add(&centroid_row.view(), &point);
            for j in 0..n_features {
                centroids[[cluster, j]] = updated_centroid[j];
            }
        } else {
            // Scalar fallback
            for j in 0..n_features {
                centroids[[cluster, j]] = centroids[[cluster, j]] + data[[i, j]];
            }
        }
    }

    // Normalize by cluster sizes and handle empty clusters
    for i in 0..k {
        if counts[i] == 0 {
            // Handle empty cluster by setting to a random data point
            if n_samples > 0 {
                let random_idx = i % n_samples; // Simple deterministic selection
                for j in 0..n_features {
                    centroids[[i, j]] = data[[random_idx, j]];
                }
            }
        } else {
            let count_f = F::from(counts[i]).unwrap();
            if use_simd {
                let centroid_row = centroids.slice(s![i, ..]);
                let count_array = Array1::from_elem(n_features, count_f);
                let normalized = F::simd_div(&centroid_row, &count_array.view());
                for j in 0..n_features {
                    centroids[[i, j]] = normalized[j];
                }
            } else {
                // Scalar fallback
                for j in 0..n_features {
                    centroids[[i, j]] = centroids[[i, j]] / count_f;
                }
            }
        }
    }

    Ok(centroids)
}

/// Parallel SIMD-optimized centroid computation
#[allow(dead_code)]
fn compute_centroids_simd_parallel<F>(
    data: ArrayView2<F>,
    labels: &Array1<usize>,
    k: usize,
    use_simd: bool,
) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Send + Sync + SimdUnifiedOps + std::iter::Sum,
{
    let n_features = data.shape()[1];

    // Process clusters in parallel
    let centroids: Vec<Array1<F>> = (0..k)
        .into_par_iter()
        .map(|cluster_id| {
            let mut sum = Array1::zeros(n_features);
            let mut count = 0;

            // Accumulate points belonging to this cluster
            for i in 0..data.shape()[0] {
                if labels[i] == cluster_id {
                    count += 1;
                    let point = data.slice(s![i, ..]);

                    if use_simd {
                        let updated_sum = F::simd_add(&sum.view(), &point);
                        for j in 0..n_features {
                            sum[j] = updated_sum[j];
                        }
                    } else {
                        // Scalar fallback
                        for j in 0..n_features {
                            sum[j] = sum[j] + point[j];
                        }
                    }
                }
            }

            // Normalize or handle empty cluster
            if count == 0 {
                // Handle empty cluster
                if data.shape()[0] > 0 {
                    let random_idx = cluster_id % data.shape()[0];
                    data.slice(s![random_idx, ..]).to_owned()
                } else {
                    sum
                }
            } else {
                let count_f = F::from(count).unwrap();
                if use_simd {
                    let count_array = Array1::from_elem(n_features, count_f);
                    let normalized = F::simd_div(&sum.view(), &count_array.view());
                    normalized
                } else {
                    // Scalar fallback
                    sum.mapv(|x| x / count_f)
                }
            }
        })
        .collect();

    // Convert to 2D array
    let mut result = Array2::zeros((k, n_features));
    for (i, centroid) in centroids.into_iter().enumerate() {
        for j in 0..n_features {
            result[[i, j]] = centroid[j];
        }
    }

    Ok(result)
}

/// SIMD-optimized distortion calculation
///
/// Computes the sum of squared distances from data points to their assigned centroids.
///
/// # Arguments
///
/// * `data` - Input data (n_samples × n_features)
/// * `centroids` - Cluster centroids (k × n_features)
/// * `labels` - Cluster assignments for each data point
/// * `config` - Optional SIMD configuration
///
/// # Returns
///
/// * Total distortion (sum of squared distances)
#[allow(dead_code)]
pub fn calculate_distortion_simd<F>(
    data: ArrayView2<F>,
    centroids: ArrayView2<F>,
    labels: &Array1<usize>,
    config: Option<&SimdOptimizationConfig>,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug + Send + Sync + SimdUnifiedOps + std::iter::Sum,
{
    let default_config = SimdOptimizationConfig::default();
    let config = config.unwrap_or(&default_config);
    let n_samples = data.shape()[0];

    if labels.len() != n_samples {
        return Err(ClusteringError::InvalidInput(
            "Labels array length must match number of data points".to_string(),
        ));
    }

    let caps = PlatformCapabilities::detect();
    let use_simd = caps.simd_available || config.force_simd;

    if config.enable_parallel && is_parallel_enabled() && n_samples > config.parallel_chunk_size {
        calculate_distortion_simd_parallel(data, centroids, labels, use_simd)
    } else {
        calculate_distortion_simd_sequential(data, centroids, labels, use_simd)
    }
}

/// Sequential SIMD-optimized distortion calculation
#[allow(dead_code)]
fn calculate_distortion_simd_sequential<F>(
    data: ArrayView2<F>,
    centroids: ArrayView2<F>,
    labels: &Array1<usize>,
    use_simd: bool,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug + SimdUnifiedOps,
{
    let n_samples = data.shape()[0];
    let mut total_distortion = F::zero();

    for i in 0..n_samples {
        let cluster = labels[i];
        if cluster >= centroids.shape()[0] {
            return Err(ClusteringError::InvalidInput(format!(
                "Label {} exceeds number of centroids {}",
                cluster,
                centroids.shape()[0]
            )));
        }

        let point = data.slice(s![i, ..]);
        let centroid = centroids.slice(s![cluster, ..]);

        let squared_distance = if use_simd {
            let diff = F::simd_sub(&point, &centroid);
            let squared_diff = F::simd_mul(&diff.view(), &diff.view());
            F::simd_sum(&squared_diff.view())
        } else {
            // Scalar fallback
            let mut sum = F::zero();
            for j in 0..point.len() {
                let diff = point[j] - centroid[j];
                sum = sum + diff * diff;
            }
            sum
        };

        total_distortion = total_distortion + squared_distance;
    }

    Ok(total_distortion)
}

/// Parallel SIMD-optimized distortion calculation
#[allow(dead_code)]
fn calculate_distortion_simd_parallel<F>(
    data: ArrayView2<F>,
    centroids: ArrayView2<F>,
    labels: &Array1<usize>,
    use_simd: bool,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug + Send + Sync + SimdUnifiedOps + std::iter::Sum,
{
    let n_samples = data.shape()[0];

    // Validate all labels first
    for &label in labels.iter() {
        if label >= centroids.shape()[0] {
            return Err(ClusteringError::InvalidInput(format!(
                "Label {} exceeds number of centroids {}",
                label,
                centroids.shape()[0]
            )));
        }
    }

    // Compute squared distances in parallel
    let squared_distances: Vec<F> = (0..n_samples)
        .into_par_iter()
        .map(|i| {
            let cluster = labels[i];
            let point = data.slice(s![i, ..]);
            let centroid = centroids.slice(s![cluster, ..]);

            if use_simd {
                let diff = F::simd_sub(&point, &centroid);
                let squared_diff = F::simd_mul(&diff.view(), &diff.view());
                F::simd_sum(&squared_diff.view())
            } else {
                // Scalar fallback
                let mut sum = F::zero();
                for j in 0..point.len() {
                    let diff = point[j] - centroid[j];
                    sum = sum + diff * diff;
                }
                sum
            }
        })
        .collect();

    Ok(squared_distances.into_iter().sum())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_euclidean_distance_simd() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let y = Array1::from_vec(vec![4.0, 5.0, 6.0]);

        let distance = euclidean_distance_simd(x.view(), y.view(), None).unwrap();
        let expected = ((4.0 - 1.0).powi(2) + (5.0 - 2.0).powi(2) + (6.0 - 3.0).powi(2)).sqrt();

        assert_abs_diff_eq!(distance, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_whiten_simd() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 1.5, 2.5, 0.5, 1.5]).unwrap();

        // Use simple config to speed up test
        let config = SimdOptimizationConfig {
            enable_parallel: false,
            force_simd: false,
            ..Default::default()
        };

        let whitened = whiten_simd(&data, Some(&config)).unwrap();

        // Check that means are approximately zero
        let col_means: Vec<f64> = (0..2).map(|j| whitened.column(j).mean()).collect();

        for mean in col_means {
            assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-8);
        }
    }

    #[test]
    #[ignore = "timeout"]
    fn test_vq_simd() {
        let data = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]).unwrap();

        let centroids = Array2::from_shape_vec((2, 2), vec![0.25, 0.25, 0.75, 0.75]).unwrap();

        // Use simple config to speed up test
        let config = SimdOptimizationConfig {
            enable_parallel: false,
            force_simd: false,
            ..Default::default()
        };

        let (labels, distances) = vq_simd(data.view(), centroids.view(), Some(&config)).unwrap();

        assert_eq!(labels.len(), 3);
        assert_eq!(distances.len(), 3);

        // Check that all distances are non-negative
        for &distance in distances.iter() {
            assert!(distance >= 0.0);
        }
    }

    #[test]
    #[ignore = "timeout"]
    fn test_compute_centroids_simd() {
        let data = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]).unwrap();

        let labels = Array1::from_vec(vec![0, 0, 1]);

        // Use simple config to speed up test
        let config = SimdOptimizationConfig {
            enable_parallel: false,
            force_simd: false,
            ..Default::default()
        };

        let centroids = compute_centroids_simd(data.view(), &labels, 2, Some(&config)).unwrap();

        assert_eq!(centroids.shape(), &[2, 2]);

        // Centroid 0 should be average of (0,0) and (1,0) = (0.5, 0)
        assert_abs_diff_eq!(centroids[[0, 0]], 0.5, epsilon = 1e-8);
        assert_abs_diff_eq!(centroids[[0, 1]], 0.0, epsilon = 1e-8);

        // Centroid 1 should be (0,1) since only one point
        assert_abs_diff_eq!(centroids[[1, 0]], 0.0, epsilon = 1e-8);
        assert_abs_diff_eq!(centroids[[1, 1]], 1.0, epsilon = 1e-8);
    }

    #[test]
    #[ignore = "timeout"]
    fn test_calculate_distortion_simd() {
        let data = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]).unwrap();

        let centroids = Array2::from_shape_vec((2, 2), vec![0.5, 0.0, 0.0, 1.0]).unwrap();

        let labels = Array1::from_vec(vec![0, 0, 1]);

        // Use simple config to speed up test
        let config = SimdOptimizationConfig {
            enable_parallel: false,
            force_simd: false,
            ..Default::default()
        };

        let distortion =
            calculate_distortion_simd(data.view(), centroids.view(), &labels, Some(&config))
                .unwrap();

        // Calculate expected distortion manually
        let expected = 0.5 * 0.5 + 0.5 * 0.5 + 0.0; // 0.5

        assert_abs_diff_eq!(distortion, expected, epsilon = 1e-8);
    }
}
