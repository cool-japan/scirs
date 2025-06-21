//! SIMD-accelerated distance computations for clustering algorithms
//!
//! This module provides highly optimized distance calculations using SIMD instructions
//! where available, with fallbacks to standard implementations.

use ndarray::{Array1, Array2, ArrayView2};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Compute Euclidean distances between all pairs of points using SIMD when available
///
/// # Arguments
///
/// * `data` - Input data (n_samples × n_features)
///
/// # Returns
///
/// * Condensed distance matrix as a 1D array
pub fn pairwise_euclidean_simd<F>(data: ArrayView2<F>) -> Array1<F>
where
    F: Float + FromPrimitive + Debug + Send + Sync,
{
    let n_samples = data.shape()[0];
    let _n_features = data.shape()[1];
    let n_distances = n_samples * (n_samples - 1) / 2;

    let mut distances = Array1::zeros(n_distances);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && std::mem::size_of::<F>() == 8 {
            // Use AVX2 for f64
            unsafe {
                pairwise_euclidean_avx2_f64(data, &mut distances);
            }
            return distances;
        } else if is_x86_feature_detected!("sse2") && std::mem::size_of::<F>() == 8 {
            // Use SSE2 for f64
            unsafe {
                pairwise_euclidean_sse2_f64(data, &mut distances);
            }
            return distances;
        }
    }

    // Fallback to standard implementation
    pairwise_euclidean_standard(data, &mut distances);
    distances
}

/// Standard pairwise Euclidean distance computation
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pairwise_euclidean_avx2_f64<F>(data: ArrayView2<F>, distances: &mut Array1<F>)
where
    F: Float + FromPrimitive + Debug,
{
    // This is a simplified implementation that assumes F is f64
    // In a real implementation, you would need proper type handling
    if std::mem::size_of::<F>() != 8 {
        pairwise_euclidean_standard(data, distances);
        return;
    }

    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];

    // AVX2 processes 4 f64 values at once
    let simd_width = 4;
    let simd_features = (n_features / simd_width) * simd_width;

    let mut idx = 0;
    for i in 0..n_samples {
        for j in (i + 1)..n_samples {
            let mut sum_vec = _mm256_setzero_pd();

            // Process features in chunks of 4
            let mut k = 0;
            while k < simd_features {
                // Load data points
                let ptr_i = data.as_ptr().add(i * n_features + k) as *const f64;
                let ptr_j = data.as_ptr().add(j * n_features + k) as *const f64;

                let vec_i = _mm256_loadu_pd(ptr_i);
                let vec_j = _mm256_loadu_pd(ptr_j);

                // Compute difference
                let diff = _mm256_sub_pd(vec_i, vec_j);

                // Square the differences and accumulate
                let diff_sq = _mm256_mul_pd(diff, diff);
                sum_vec = _mm256_add_pd(sum_vec, diff_sq);

                k += simd_width;
            }

            // Extract the sum from the SIMD register
            let mut sum_array = [0.0; 4];
            _mm256_storeu_pd(sum_array.as_mut_ptr(), sum_vec);
            let mut sum_sq = sum_array.iter().sum::<f64>();

            // Handle remaining features
            for k in simd_features..n_features {
                let val_i = *((data.as_ptr().add(i * n_features + k)) as *const f64);
                let val_j = *((data.as_ptr().add(j * n_features + k)) as *const f64);
                let diff = val_i - val_j;
                sum_sq += diff * diff;
            }

            distances[idx] = F::from_f64(sum_sq.sqrt()).unwrap();
            idx += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn pairwise_euclidean_sse2_f64<F>(data: ArrayView2<F>, distances: &mut Array1<F>)
where
    F: Float + FromPrimitive + Debug,
{
    // Similar to AVX2 but using SSE2 (2 f64 values at once)
    if std::mem::size_of::<F>() != 8 {
        pairwise_euclidean_standard(data, distances);
        return;
    }

    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];

    // SSE2 processes 2 f64 values at once
    let simd_width = 2;
    let simd_features = (n_features / simd_width) * simd_width;

    let mut idx = 0;
    for i in 0..n_samples {
        for j in (i + 1)..n_samples {
            let mut sum_vec = _mm_setzero_pd();

            // Process features in chunks of 2
            let mut k = 0;
            while k < simd_features {
                let ptr_i = data.as_ptr().add(i * n_features + k) as *const f64;
                let ptr_j = data.as_ptr().add(j * n_features + k) as *const f64;

                let vec_i = _mm_loadu_pd(ptr_i);
                let vec_j = _mm_loadu_pd(ptr_j);

                let diff = _mm_sub_pd(vec_i, vec_j);
                let diff_sq = _mm_mul_pd(diff, diff);
                sum_vec = _mm_add_pd(sum_vec, diff_sq);

                k += simd_width;
            }

            // Extract the sum
            let mut sum_array = [0.0; 2];
            _mm_storeu_pd(sum_array.as_mut_ptr(), sum_vec);
            let mut sum_sq = sum_array.iter().sum::<f64>();

            // Handle remaining features
            for k in simd_features..n_features {
                let val_i = *((data.as_ptr().add(i * n_features + k)) as *const f64);
                let val_j = *((data.as_ptr().add(j * n_features + k)) as *const f64);
                let diff = val_i - val_j;
                sum_sq += diff * diff;
            }

            distances[idx] = F::from_f64(sum_sq.sqrt()).unwrap();
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
pub fn distance_to_centroids_simd<F>(data: ArrayView2<F>, centroids: ArrayView2<F>) -> Array2<F>
where
    F: Float + FromPrimitive + Debug + Send + Sync,
{
    let n_samples = data.shape()[0];
    let n_clusters = centroids.shape()[0];
    let n_features = data.shape()[1];

    if centroids.shape()[1] != n_features {
        panic!("Data and centroids must have the same number of features");
    }

    let mut distances = Array2::zeros((n_samples, n_clusters));

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && std::mem::size_of::<F>() == 8 {
            unsafe {
                distance_to_centroids_avx2_f64(data, centroids, &mut distances);
            }
            return distances;
        }
    }

    // Fallback to standard implementation
    distance_to_centroids_standard(data, centroids, &mut distances);
    distances
}

/// Standard distance to centroids computation
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn distance_to_centroids_avx2_f64<F>(
    data: ArrayView2<F>,
    centroids: ArrayView2<F>,
    distances: &mut Array2<F>,
) where
    F: Float + FromPrimitive + Debug,
{
    if std::mem::size_of::<F>() != 8 {
        distance_to_centroids_standard(data, centroids, distances);
        return;
    }

    let n_samples = data.shape()[0];
    let n_clusters = centroids.shape()[0];
    let n_features = data.shape()[1];

    let simd_width = 4;
    let simd_features = (n_features / simd_width) * simd_width;

    for i in 0..n_samples {
        for j in 0..n_clusters {
            let mut sum_vec = _mm256_setzero_pd();

            let mut k = 0;
            while k < simd_features {
                let ptr_data = data.as_ptr().add(i * n_features + k) as *const f64;
                let ptr_centroid = centroids.as_ptr().add(j * n_features + k) as *const f64;

                let vec_data = _mm256_loadu_pd(ptr_data);
                let vec_centroid = _mm256_loadu_pd(ptr_centroid);

                let diff = _mm256_sub_pd(vec_data, vec_centroid);
                let diff_sq = _mm256_mul_pd(diff, diff);
                sum_vec = _mm256_add_pd(sum_vec, diff_sq);

                k += simd_width;
            }

            let mut sum_array = [0.0; 4];
            _mm256_storeu_pd(sum_array.as_mut_ptr(), sum_vec);
            let mut sum_sq = sum_array.iter().sum::<f64>();

            // Handle remaining features
            for k in simd_features..n_features {
                let val_data = *((data.as_ptr().add(i * n_features + k)) as *const f64);
                let val_centroid = *((centroids.as_ptr().add(j * n_features + k)) as *const f64);
                let diff = val_data - val_centroid;
                sum_sq += diff * diff;
            }

            distances[[i, j]] = F::from_f64(sum_sq.sqrt()).unwrap();
        }
    }
}

/// Parallel distance matrix computation using Rayon
///
/// # Arguments
///
/// * `data` - Input data (n_samples × n_features)
///
/// # Returns
///
/// * Condensed distance matrix
#[cfg(feature = "rayon")]
pub fn pairwise_euclidean_parallel<F>(data: ArrayView2<F>) -> Array1<F>
where
    F: Float + FromPrimitive + Debug + Send + Sync,
{
    use rayon::prelude::*;

    let n_samples = data.shape()[0];
    let n_distances = n_samples * (n_samples - 1) / 2;

    // Create index pairs
    let mut pairs = Vec::with_capacity(n_distances);
    for i in 0..n_samples {
        for j in (i + 1)..n_samples {
            pairs.push((i, j));
        }
    }

    // Compute distances in parallel
    let distances: Vec<F> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let mut sum_sq = F::zero();
            for k in 0..data.shape()[1] {
                let diff = data[[i, k]] - data[[j, k]];
                sum_sq = sum_sq + diff * diff;
            }
            sum_sq.sqrt()
        })
        .collect();

    Array1::from_vec(distances)
}

#[cfg(not(feature = "rayon"))]
pub fn pairwise_euclidean_parallel<F>(data: ArrayView2<F>) -> Array1<F>
where
    F: Float + FromPrimitive + Debug + Send + Sync,
{
    // Fallback to SIMD version when rayon is not available
    pairwise_euclidean_simd(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array2;

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

        let distances = distance_to_centroids_simd(data.view(), centroids.view());

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
