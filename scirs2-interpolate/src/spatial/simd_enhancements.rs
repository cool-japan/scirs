//! Additional SIMD enhancements for spatial search operations
//!
//! This module provides enhanced SIMD operations that complement the existing
//! optimized_search.rs functionality with more specialized optimizations.

use scirs2_core::ndarray::{Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::numeric::{Float, Zero};

#[cfg(all(feature = "simd", test))]
use scirs2_core::ndarray::Array1;

#[cfg(feature = "simd")]
use scirs2_core::numeric::FromPrimitive;

#[cfg(feature = "simd")]
use std::fmt::Debug;

#[cfg(feature = "simd")]
use crate::error::InterpolateResult;

#[cfg(feature = "simd")]
use scirs2_core::simd_ops::SimdUnifiedOps;

#[cfg(feature = "parallel")]
use scirs2_core::parallel_ops::*;

/// Advanced SIMD-optimized spatial operations
pub struct AdvancedSimdOps;

impl AdvancedSimdOps {
    /// Vectorized distance computation with multiple query points to multiple data points
    #[cfg(feature = "simd")]
    pub fn vectorized_distance_matrix<F>(
        queries: &ArrayView2<F>,
        points: &ArrayView2<F>,
    ) -> Array2<F>
    where
        F: Float + FromPrimitive + SimdUnifiedOps + Zero + Send + Sync + Debug,
    {
        let n_queries = queries.nrows();
        let n_points = points.nrows();
        let dim = queries.ncols();

        assert_eq!(points.ncols(), dim, "Query and point dimensions must match");

        let mut distance_matrix = Array2::zeros((n_queries, n_points));

        if dim >= 4 {
            // Use SIMD for vector operations
            for (q_idx, query) in queries.axis_iter(Axis(0)).enumerate() {
                for (p_idx, point) in points.axis_iter(Axis(0)).enumerate() {
                    // Vectorized distance computation
                    let diff = F::simd_sub(&query, &point);
                    let squared = F::simd_mul(&diff.view(), &diff.view());
                    let distance = F::simd_sum(&squared.view());
                    distance_matrix[[q_idx, p_idx]] = distance;
                }
            }
        } else {
            // Scalar fallback for low dimensions
            for (q_idx, query) in queries.axis_iter(Axis(0)).enumerate() {
                for (p_idx, point) in points.axis_iter(Axis(0)).enumerate() {
                    let distance = query
                        .iter()
                        .zip(point.iter())
                        .map(|(&q, &p)| {
                            let diff = q - p;
                            diff * diff
                        })
                        .fold(F::zero(), |acc, x| acc + x);
                    distance_matrix[[q_idx, p_idx]] = distance;
                }
            }
        }

        distance_matrix
    }

    /// Non-SIMD fallback for vectorized distance computation
    #[cfg(not(feature = "simd"))]
    pub fn vectorized_distance_matrix<F>(
        queries: &ArrayView2<F>,
        points: &ArrayView2<F>,
    ) -> Array2<F>
    where
        F: Float + Zero + Send + Sync + std::fmt::Debug,
    {
        let n_queries = queries.nrows();
        let n_points = points.nrows();
        let dim = queries.ncols();

        assert_eq!(points.ncols(), dim, "Query and point dimensions must match");

        let mut distance_matrix = Array2::zeros((n_queries, n_points));

        // Scalar fallback for all dimensions
        for (q_idx, query) in queries.axis_iter(Axis(0)).enumerate() {
            for (p_idx, point) in points.axis_iter(Axis(0)).enumerate() {
                let distance = query
                    .iter()
                    .zip(point.iter())
                    .map(|(&q, &p)| {
                        let diff = q - p;
                        diff * diff
                    })
                    .fold(F::zero(), |acc, x| acc + x);
                distance_matrix[[q_idx, p_idx]] = distance;
            }
        }

        distance_matrix
    }

    /// Parallel SIMD-optimized k-nearest neighbor search for batch queries
    #[cfg(all(feature = "simd", feature = "parallel"))]
    pub fn parallel_batch_knn<F>(
        points: &ArrayView2<F>,
        queries: &ArrayView2<F>,
        k: usize,
    ) -> Vec<Vec<(usize, F)>>
    where
        F: Float
            + FromPrimitive
            + SimdUnifiedOps
            + Zero
            + Send
            + Sync
            + Debug
            + PartialOrd
            + ordered_float::FloatCore,
    {
        let n_queries = queries.nrows();

        // Use parallel processing for large query sets
        if n_queries >= 4 {
            (0..n_queries)
                .into_par_iter()
                .map(|query_idx| {
                    let query = queries.row(query_idx);
                    Self::simd_single_knn(points, &query, k)
                })
                .collect()
        } else {
            // Sequential processing for small query sets
            queries
                .axis_iter(Axis(0))
                .map(|query| Self::simd_single_knn(points, &query, k))
                .collect()
        }
    }

    /// SIMD-optimized single query k-nearest neighbor search
    #[cfg(feature = "simd")]
    pub fn simd_single_knn<F>(
        points: &ArrayView2<F>,
        query: &ArrayView1<F>,
        k: usize,
    ) -> Vec<(usize, F)>
    where
        F: Float
            + FromPrimitive
            + SimdUnifiedOps
            + Zero
            + Debug
            + PartialOrd
            + ordered_float::FloatCore,
    {
        let n_points = points.nrows();
        let dim = points.ncols();

        assert_eq!(
            query.len(),
            dim,
            "Query dimension must match point dimension"
        );

        if k >= n_points {
            // Return all points if k is larger than available points
            let mut all_distances: Vec<(usize, F)> = (0..n_points)
                .map(|idx| {
                    let point = points.row(idx);
                    let distance = if dim >= 4 {
                        let diff = F::simd_sub(query, &point);
                        let squared = F::simd_mul(&diff.view(), &diff.view());
                        F::simd_sum(&squared.view())
                    } else {
                        Self::scalar_distance(query, &point)
                    };
                    (idx, distance)
                })
                .collect();

            all_distances
                .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            return all_distances;
        }

        // Use a max-heap to maintain k nearest neighbors
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let mut heap: BinaryHeap<Reverse<(ordered_float::OrderedFloat<F>, usize)>> =
            BinaryHeap::with_capacity(k + 1);

        // Process points in chunks for better cache locality
        const CHUNK_SIZE: usize = 64;

        for chunk_start in (0..n_points).step_by(CHUNK_SIZE) {
            let chunk_end = (chunk_start + CHUNK_SIZE).min(n_points);

            for point_idx in chunk_start..chunk_end {
                let point = points.row(point_idx);

                let distance = if dim >= 4 {
                    // Use SIMD for higher dimensions
                    let diff = F::simd_sub(query, &point);
                    let squared = F::simd_mul(&diff.view(), &diff.view());
                    F::simd_sum(&squared.view())
                } else {
                    // Scalar computation for low dimensions
                    Self::scalar_distance(query, &point)
                };

                if heap.len() < k {
                    heap.push(Reverse((ordered_float::OrderedFloat(distance), point_idx)));
                } else if let Some(Reverse((max_dist_, _))) = heap.peek() {
                    if ordered_float::OrderedFloat(distance) < *max_dist_ {
                        heap.pop();
                        heap.push(Reverse((ordered_float::OrderedFloat(distance), point_idx)));
                    }
                }
            }
        }

        // Convert heap to sorted vector
        let mut result: Vec<(usize, F)> = heap
            .into_iter()
            .map(|Reverse((dist, idx))| (idx, dist.into_inner()))
            .collect();

        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    /// Scalar distance computation fallback
    #[allow(dead_code)]
    fn scalar_distance<F>(a: &ArrayView1<F>, b: &ArrayView1<F>) -> F
    where
        F: Float + Zero,
    {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let diff = x - y;
                diff * diff
            })
            .fold(F::zero(), |acc, x| acc + x)
    }

    /// SIMD-optimized range search (all points within radius)
    #[cfg(feature = "simd")]
    pub fn simd_range_search<F>(
        points: &ArrayView2<F>,
        query: &ArrayView1<F>,
        radius_squared: F,
    ) -> Vec<(usize, F)>
    where
        F: Float
            + FromPrimitive
            + SimdUnifiedOps
            + Zero
            + Debug
            + PartialOrd
            + ordered_float::FloatCore,
    {
        let n_points = points.nrows();
        let dim = points.ncols();

        assert_eq!(
            query.len(),
            dim,
            "Query dimension must match point dimension"
        );

        let mut results = Vec::new();

        if dim >= 4 {
            // SIMD processing
            for point_idx in 0..n_points {
                let point = points.row(point_idx);
                let diff = F::simd_sub(query, &point);
                let _squared = F::simd_mul(&diff.view(), &diff.view());
                let distance_sq = F::simd_sum(&_squared.view());

                if distance_sq <= radius_squared {
                    results.push((point_idx, distance_sq));
                }
            }
        } else {
            // Scalar processing
            for point_idx in 0..n_points {
                let point = points.row(point_idx);
                let distance_sq = Self::scalar_distance(query, &point);

                if distance_sq <= radius_squared {
                    results.push((point_idx, distance_sq));
                }
            }
        }

        // Sort by distance
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Adaptive SIMD strategy selection based on problem characteristics
    #[cfg(feature = "simd")]
    pub fn adaptive_simd_strategy<F>(
        points: &ArrayView2<F>,
        queries: &ArrayView2<F>,
        k: usize,
    ) -> InterpolateResult<Vec<Vec<(usize, F)>>>
    where
        F: Float
            + FromPrimitive
            + SimdUnifiedOps
            + Zero
            + Send
            + Sync
            + Debug
            + PartialOrd
            + ordered_float::FloatCore,
    {
        let n_points = points.nrows();
        let n_queries = queries.nrows();
        let dim = points.ncols();

        // Choose strategy based on problem characteristics
        if n_queries >= 8 && n_points >= 100 && dim >= 4 {
            // Large problem: use parallel SIMD
            #[cfg(feature = "parallel")]
            {
                Ok(Self::parallel_batch_knn(points, queries, k))
            }
            #[cfg(not(feature = "parallel"))]
            {
                // Fallback to sequential SIMD
                Ok(queries
                    .axis_iter(Axis(0))
                    .map(|query| Self::simd_single_knn(points, &query, k))
                    .collect())
            }
        } else if dim >= 4 {
            // Medium problem: use sequential SIMD
            Ok(queries
                .axis_iter(Axis(0))
                .map(|query| Self::simd_single_knn(points, &query, k))
                .collect())
        } else {
            // Small problem or low dimension: use scalar computation
            Ok(queries
                .axis_iter(Axis(0))
                .map(|query| Self::scalar_knn(points, &query, k))
                .collect())
        }
    }

    /// Scalar k-nearest neighbor search for comparison and fallback
    #[allow(dead_code)]
    fn scalar_knn<F>(points: &ArrayView2<F>, query: &ArrayView1<F>, k: usize) -> Vec<(usize, F)>
    where
        F: Float + Zero + PartialOrd,
    {
        let n_points = points.nrows();
        let k = k.min(n_points);

        let mut distances: Vec<(usize, F)> = (0..n_points)
            .map(|idx| {
                let point = points.row(idx);
                let distance = Self::scalar_distance(query, &point);
                (idx, distance)
            })
            .collect();

        // Use partial sort for efficiency when k << n
        if k < n_points / 4 {
            distances.select_nth_unstable_by(k, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            distances.truncate(k);
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            distances.truncate(k);
        }

        distances
    }

    /// Memory-efficient streaming k-NN search for very large datasets
    #[cfg(feature = "simd")]
    pub fn streaming_knn<F>(
        points: &ArrayView2<F>,
        query: &ArrayView1<F>,
        k: usize,
        chunk_size: usize,
    ) -> Vec<(usize, F)>
    where
        F: Float
            + FromPrimitive
            + SimdUnifiedOps
            + Zero
            + Debug
            + PartialOrd
            + ordered_float::FloatCore,
    {
        let n_points = points.nrows();
        let dim = points.ncols();

        // Use a min-heap to track k best candidates
        use std::collections::BinaryHeap;
        let mut heap: BinaryHeap<(ordered_float::OrderedFloat<F>, usize)> =
            BinaryHeap::with_capacity(k + 1);

        // Process points in chunks to maintain memory efficiency
        for chunk_start in (0..n_points).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(n_points);

            for point_idx in chunk_start..chunk_end {
                let point = points.row(point_idx);

                let distance = if dim >= 4 {
                    let diff = F::simd_sub(query, &point);
                    let squared = F::simd_mul(&diff.view(), &diff.view());
                    F::simd_sum(&squared.view())
                } else {
                    Self::scalar_distance(query, &point)
                };

                if heap.len() < k {
                    heap.push((ordered_float::OrderedFloat(distance), point_idx));
                } else if let Some((max_dist_, _)) = heap.peek() {
                    if ordered_float::OrderedFloat(distance) < *max_dist_ {
                        heap.pop();
                        heap.push((ordered_float::OrderedFloat(distance), point_idx));
                    }
                }
            }
        }

        // Convert to sorted result
        let mut result: Vec<(usize, F)> = heap
            .into_iter()
            .map(|(dist, idx)| (idx, dist.into_inner()))
            .collect();

        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }
}

/// Performance benchmark utilities for SIMD optimization evaluation
pub struct SimdBenchmark;

impl SimdBenchmark {
    /// Compare SIMD vs scalar performance for different problem sizes
    #[cfg(all(feature = "simd", test))]
    pub fn benchmark_distance_computation<F>(
        points: &ArrayView2<F>,
        queries: &ArrayView2<F>,
    ) -> (std::time::Duration, std::time::Duration)
    where
        F: Float + FromPrimitive + SimdUnifiedOps + Zero + Send + Sync + Debug + PartialOrd,
    {
        use std::time::Instant;

        // SIMD timing
        let start = Instant::now();
        let _simd_results = AdvancedSimdOps::vectorized_distance_matrix(queries, points);
        let simd_time = start.elapsed();

        // Scalar timing
        let start = Instant::now();
        let mut scalar_results = Array2::zeros((queries.nrows(), points.nrows()));
        for (q_idx, query) in queries.axis_iter(Axis(0)).enumerate() {
            for (p_idx, point) in points.axis_iter(Axis(0)).enumerate() {
                let distance = AdvancedSimdOps::scalar_distance(&query, &point);
                scalar_results[[q_idx, p_idx]] = distance;
            }
        }
        let scalar_time = start.elapsed();

        (simd_time, scalar_time)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_simd_distance_computation() {
        let points = Array2::from_shape_vec(
            (4, 3),
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        )
        .unwrap();

        let queries = Array2::from_shape_vec((2, 3), vec![0.5, 0.5, 0.0, 1.0, 1.0, 0.0]).unwrap();

        let distance_matrix =
            AdvancedSimdOps::vectorized_distance_matrix(&queries.view(), &points.view());

        assert_eq!(distance_matrix.shape(), &[2, 4]);

        // Check a few known distances
        // Query [0.5, 0.5, 0.0] to point [0.0, 0.0, 0.0] should be 0.5
        assert_abs_diff_eq!(distance_matrix[[0, 0]], 0.5, epsilon = 1e-10);
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_knn_search() {
        let points = Array2::from_shape_vec(
            (5, 2),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5],
        )
        .unwrap();

        let query = Array1::from(vec![0.6, 0.6]);
        let knn = AdvancedSimdOps::simd_single_knn(&points.view(), &query.view(), 3);

        assert_eq!(knn.len(), 3);
        // The nearest point should be [0.5, 0.5] at index 4
        assert_eq!(knn[0].0, 4);
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_range_search() {
        let points = Array2::from_shape_vec(
            (5, 2),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5],
        )
        .unwrap();

        let query = Array1::from(vec![0.5, 0.5]);
        let radius_sq = 0.5; // Radius of sqrt(0.5) ≈ 0.707

        let range_results =
            AdvancedSimdOps::simd_range_search(&points.view(), &query.view(), radius_sq);

        // Should find point [0.5, 0.5] itself and potentially others within range
        assert!(!range_results.is_empty());
        assert_eq!(range_results[0].0, 4); // Point [0.5, 0.5] at index 4
        assert_abs_diff_eq!(range_results[0].1, 0.0, epsilon = 1e-10);
    }
}
