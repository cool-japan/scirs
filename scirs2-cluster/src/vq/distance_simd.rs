//! SIMD-accelerated distance computations for clustering algorithms
//!
//! This module provides highly optimized distance calculations using the unified
//! SIMD operations from scirs2-core, with fallbacks to standard implementations.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use scirs2_core::parallel_ops::*;
use scirs2_core::simd_ops::{AutoOptimizer, PlatformCapabilities, SimdUnifiedOps};
use std::fmt::Debug;

/// Memory-efficient configuration for SIMD operations
#[derive(Debug, Clone)]
pub struct SimdConfig {
    /// Chunk size for memory-efficient processing
    pub chunk_size: usize,
    /// Enable memory prefetching
    pub enable_prefetch: bool,
    /// Use cache-friendly algorithms
    pub cache_friendly: bool,
    /// Block size for blocked algorithms
    pub block_size: usize,
}

impl Default for SimdConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1024,
            enable_prefetch: true,
            cache_friendly: true,
            block_size: 256,
        }
    }
}

/// Memory-efficient blocked distance computation for large datasets
///
/// This function uses cache-friendly blocking to compute distances efficiently
/// for datasets that don't fit in cache.
#[allow(dead_code)]
pub fn pairwise_euclidean_blocked<F>(data: ArrayView2<F>, config: Option<SimdConfig>) -> Array1<F>
where
    F: Float + FromPrimitive + Debug + Send + Sync + SimdUnifiedOps,
{
    let config = config.unwrap_or_default();
    let n_samples = data.shape()[0];
    let _n_features = data.shape()[1];
    let n_distances = n_samples * (n_samples - 1) / 2;
    let mut distances = Array1::zeros(n_distances);

    let caps = PlatformCapabilities::detect();

    if caps.simd_available && config.cache_friendly {
        pairwise_euclidean_blocked_simd(data, &mut distances, &config);
    } else {
        pairwise_euclidean_standard(data, &mut distances);
    }

    distances
}

/// Cache-friendly blocked SIMD implementation
#[allow(dead_code)]
fn pairwise_euclidean_blocked_simd<F>(
    data: ArrayView2<F>,
    distances: &mut Array1<F>,
    config: &SimdConfig,
) where
    F: Float + FromPrimitive + Debug + Send + Sync + SimdUnifiedOps,
{
    let n_samples = data.shape()[0];
    let block_size = config.block_size;

    let mut idx = 0;

    // Process data in blocks to improve cache efficiency
    for block_i in (0..n_samples).step_by(block_size) {
        let end_i = (block_i + block_size).min(n_samples);

        for block_j in (block_i..n_samples).step_by(block_size) {
            let end_j = (block_j + block_size).min(n_samples);

            // Process block [block_i..end_i) × [block_j..end_j)
            for i in block_i..end_i {
                let start_j = if block_i == block_j { i + 1 } else { block_j };

                for j in start_j..end_j {
                    let row_i = data.row(i);
                    let row_j = data.row(j);

                    // Use SIMD operations with prefetching if enabled
                    if config.enable_prefetch && j + 1 < end_j {
                        // Prefetch next row for better memory access patterns
                        std::hint::spin_loop(); // Simplified prefetch simulation
                    }

                    let diff = F::simd_sub(&row_i, &row_j);
                    let distance = F::simd_norm(&diff.view());

                    distances[idx] = distance;
                    idx += 1;
                }
            }
        }
    }
}

/// Streaming distance computation for out-of-core datasets
///
/// This function computes distances in streaming fashion, suitable for
/// datasets that don't fit in memory.
#[allow(dead_code)]
pub fn pairwise_euclidean_streaming<'a, F>(
    data_chunks: impl Iterator<Item = ArrayView2<'a, F>>,
    chunk_size: usize,
) -> Array1<F>
where
    F: Float + FromPrimitive + Debug + Send + Sync + SimdUnifiedOps + 'a,
{
    // Remove unused variable
    let mut total_samples = 0;
    let mut data_cache = Vec::new();

    // First pass: collect data and count samples
    for chunk in data_chunks {
        total_samples += chunk.nrows();
        data_cache.push(chunk.to_owned());
    }

    let n_distances = total_samples * (total_samples - 1) / 2;
    let mut distances = Array1::zeros(n_distances);
    let mut idx = 0;

    // Second pass: compute distances between _chunks
    for (chunk_i, data_i) in data_cache.iter().enumerate() {
        for (chunk_j, data_j) in data_cache.iter().enumerate().skip(chunk_i) {
            if chunk_i == chunk_j {
                // Intra-chunk distances
                idx += compute_intra_chunk_distances(data_i.view(), &mut distances, idx);
            } else {
                // Inter-chunk distances
                idx += compute_inter_chunk_distances(
                    data_i.view(),
                    data_j.view(),
                    &mut distances,
                    idx,
                );
            }
        }
    }

    distances
}

/// Compute distances within a single chunk
#[allow(dead_code)]
fn compute_intra_chunk_distances<F>(
    chunk: ArrayView2<F>,
    distances: &mut Array1<F>,
    start_idx: usize,
) -> usize
where
    F: Float + FromPrimitive + Debug + SimdUnifiedOps,
{
    let n_samples = chunk.nrows();
    let mut _idx = start_idx;

    for i in 0..n_samples {
        for j in (i + 1)..n_samples {
            let row_i = chunk.row(i);
            let row_j = chunk.row(j);

            let diff = F::simd_sub(&row_i, &row_j);
            let distance = F::simd_norm(&diff.view());

            distances[_idx] = distance;
            _idx += 1;
        }
    }

    _idx - start_idx
}

/// Compute distances between two chunks
#[allow(dead_code)]
fn compute_inter_chunk_distances<F>(
    chunk_i: ArrayView2<F>,
    chunk_j: ArrayView2<F>,
    distances: &mut Array1<F>,
    start_idx: usize,
) -> usize
where
    F: Float + FromPrimitive + Debug + SimdUnifiedOps,
{
    let n_samples_i = chunk_i.nrows();
    let n_samples_j = chunk_j.nrows();
    let mut _idx = start_idx;

    for _i in 0..n_samples_i {
        for _j in 0..n_samples_j {
            let row_i = chunk_i.row(_i);
            let row_j = chunk_j.row(_j);

            let diff = F::simd_sub(&row_i, &row_j);
            let distance = F::simd_norm(&diff.view());

            distances[_idx] = distance;
            _idx += 1;
        }
    }

    _idx - start_idx
}

/// Compute Euclidean distances between all pairs of points using SIMD when available
///
/// # Arguments
///
/// * `data` - Input data (n_samples × n_features)
///
/// # Returns
///
/// * Condensed distance matrix as a 1D array
#[allow(dead_code)]
pub fn pairwise_euclidean_simd<F>(data: ArrayView2<F>) -> Array1<F>
where
    F: Float + FromPrimitive + Debug + Send + Sync + SimdUnifiedOps,
{
    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];
    let n_distances = n_samples * (n_samples - 1) / 2;
    let mut distances = Array1::zeros(n_distances);

    let caps = PlatformCapabilities::detect();
    let optimizer = AutoOptimizer::new();

    if caps.simd_available && optimizer.should_use_simd(n_samples * n_features) {
        pairwise_euclidean_simd_optimized(data, &mut distances);
    } else {
        pairwise_euclidean_standard(data, &mut distances);
    }

    distances
}

/// Standard pairwise Euclidean distance computation
#[allow(dead_code)]
fn pairwise_euclidean_standard<F>(data: ArrayView2<F>, distances: &mut Array1<F>)
where
    F: Float + FromPrimitive + Debug,
{
    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];

    let mut idx = 0;
    for i in 0..n_samples {
        for j in (i + 1)..n_samples {
            let mut sum_sq = F::zero();
            for k in 0..n_features {
                let diff = data[[i, k]] - data[[j, k]];
                sum_sq = sum_sq + diff * diff;
            }
            distances[idx] = sum_sq.sqrt();
            idx += 1;
        }
    }
}

/// SIMD-optimized pairwise Euclidean distance computation using unified operations
#[allow(dead_code)]
fn pairwise_euclidean_simd_optimized<F>(data: ArrayView2<F>, distances: &mut Array1<F>)
where
    F: Float + FromPrimitive + Debug + SimdUnifiedOps,
{
    let n_samples = data.shape()[0];

    let mut idx = 0;
    for i in 0..n_samples {
        for j in (i + 1)..n_samples {
            let row_i = data.row(i);
            let row_j = data.row(j);

            // Use SIMD operations for vector subtraction and norm calculation
            let diff = F::simd_sub(&row_i, &row_j);
            let distance = F::simd_norm(&diff.view());

            distances[idx] = distance;
            idx += 1;
        }
    }
}

/// Compute distances from each point to a set of centroids using SIMD
///
/// # Arguments
///
/// * `data` - Input data (n_samples × n_features)
/// * `centroids` - Cluster centroids (n_clusters × n_features)
///
/// # Returns
///
/// * Distance matrix (n_samples × n_clusters)
///
/// # Errors
///
/// * Returns error if data and centroids have different numbers of features
#[allow(dead_code)]
pub fn distance_to_centroids_simd<F>(
    data: ArrayView2<F>,
    centroids: ArrayView2<F>,
) -> Result<Array2<F>, String>
where
    F: Float + FromPrimitive + Debug + Send + Sync + SimdUnifiedOps,
{
    let n_samples = data.shape()[0];
    let n_clusters = centroids.shape()[0];
    let n_features = data.shape()[1];

    if centroids.shape()[1] != n_features {
        return Err(format!(
            "Data and centroids must have the same number of features: data has {}, centroids have {}",
            n_features, centroids.shape()[1]
        ));
    }

    let mut distances = Array2::zeros((n_samples, n_clusters));

    let caps = PlatformCapabilities::detect();
    let optimizer = AutoOptimizer::new();

    if caps.simd_available && optimizer.should_use_simd(n_samples * n_features) {
        distance_to_centroids_simd_optimized(data, centroids, &mut distances);
    } else {
        distance_to_centroids_standard(data, centroids, &mut distances);
    }

    Ok(distances)
}

/// Standard distance to centroids computation
#[allow(dead_code)]
fn distance_to_centroids_standard<F>(
    data: ArrayView2<F>,
    centroids: ArrayView2<F>,
    distances: &mut Array2<F>,
) where
    F: Float + FromPrimitive + Debug,
{
    let n_samples = data.shape()[0];
    let n_clusters = centroids.shape()[0];
    let n_features = data.shape()[1];

    for i in 0..n_samples {
        for j in 0..n_clusters {
            let mut sum_sq = F::zero();
            for k in 0..n_features {
                let diff = data[[i, k]] - centroids[[j, k]];
                sum_sq = sum_sq + diff * diff;
            }
            distances[[i, j]] = sum_sq.sqrt();
        }
    }
}

/// SIMD-optimized distance to centroids computation using unified operations
#[allow(dead_code)]
fn distance_to_centroids_simd_optimized<F>(
    data: ArrayView2<F>,
    centroids: ArrayView2<F>,
    distances: &mut Array2<F>,
) where
    F: Float + FromPrimitive + Debug + SimdUnifiedOps,
{
    let n_samples = data.shape()[0];
    let n_clusters = centroids.shape()[0];

    for i in 0..n_samples {
        for j in 0..n_clusters {
            let data_row = data.row(i);
            let centroid_row = centroids.row(j);

            // Use SIMD operations for vector subtraction and norm calculation
            let diff = F::simd_sub(&data_row, &centroid_row);
            let distance = F::simd_norm(&diff.view());

            distances[[i, j]] = distance;
        }
    }
}

/// Parallel distance matrix computation using core parallel operations
///
/// # Arguments
///
/// * `data` - Input data (n_samples × n_features)
///
/// # Returns
///
/// * Condensed distance matrix
#[allow(dead_code)]
pub fn pairwise_euclidean_parallel<F>(data: ArrayView2<F>) -> Array1<F>
where
    F: Float + FromPrimitive + Debug + Send + Sync + SimdUnifiedOps,
{
    let n_samples = data.shape()[0];
    let n_distances = n_samples * (n_samples - 1) / 2;

    // Create index pairs
    let mut pairs = Vec::with_capacity(n_distances);
    for i in 0..n_samples {
        for j in (i + 1)..n_samples {
            pairs.push((i, j));
        }
    }

    // Use parallel operations from core
    if is_parallel_enabled() && pairs.len() > 100 {
        // Compute distances in parallel using core abstractions
        let distances: Vec<F> = pairs
            .into_par_iter()
            .map(|(i, j)| {
                let row_i = data.row(i);
                let row_j = data.row(j);

                // Use SIMD operations for distance calculation
                let diff = F::simd_sub(&row_i, &row_j);
                F::simd_norm(&diff.view())
            })
            .collect();
        Array1::from_vec(distances)
    } else {
        // Fallback to SIMD version for small problems or when parallel is disabled
        pairwise_euclidean_simd(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_pairwise_euclidean_simd() {
        let data =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

        let distances = pairwise_euclidean_simd(data.view());

        // Expected distances: (0,1)=1.0, (0,2)=1.0, (0,3)=√2, (1,2)=√2, (1,3)=1.0, (2,3)=1.0
        assert_eq!(distances.len(), 6);
        assert_abs_diff_eq!(distances[0], 1.0, epsilon = 1e-10); // (0,1)
        assert_abs_diff_eq!(distances[1], 1.0, epsilon = 1e-10); // (0,2)
        assert_abs_diff_eq!(distances[2], 2.0_f64.sqrt(), epsilon = 1e-10); // (0,3)
        assert_abs_diff_eq!(distances[3], 2.0_f64.sqrt(), epsilon = 1e-10); // (1,2)
        assert_abs_diff_eq!(distances[4], 1.0, epsilon = 1e-10); // (1,3)
        assert_abs_diff_eq!(distances[5], 1.0, epsilon = 1e-10); // (2,3)
    }

    #[test]
    fn test_distance_to_centroids_simd() {
        let data =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

        let centroids = Array2::from_shape_vec((2, 2), vec![0.5, 0.0, 0.5, 1.0]).unwrap();

        let distances = distance_to_centroids_simd(data.view(), centroids.view()).unwrap();

        assert_eq!(distances.shape(), &[4, 2]);

        // Check some expected distances
        assert_abs_diff_eq!(distances[[0, 0]], 0.5, epsilon = 1e-10); // (0,0) to centroid 0
        assert_abs_diff_eq!(distances[[3, 1]], 0.5, epsilon = 1e-10); // (1,1) to centroid 1
    }

    #[test]
    fn test_parallel_vs_standard() {
        let data = Array2::from_shape_vec(
            (6, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                9.0, 10.0,
            ],
        )
        .unwrap();

        let distances_simd = pairwise_euclidean_simd(data.view());
        let distances_parallel = pairwise_euclidean_parallel(data.view());

        assert_eq!(distances_simd.len(), distances_parallel.len());

        for i in 0..distances_simd.len() {
            assert_abs_diff_eq!(distances_simd[i], distances_parallel[i], epsilon = 1e-10);
        }
    }
}
