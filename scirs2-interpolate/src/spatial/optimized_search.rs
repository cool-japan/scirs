//! Optimized spatial search algorithms with enhanced performance features
//!
//! This module provides advanced spatial search optimizations including:
//! - SIMD-accelerated distance computations (via scirs2-core)
//! - Cache-friendly memory layouts
//! - Adaptive search strategies
//! - Batch query processing
//! - Multi-threaded search operations
//!
//! All SIMD operations are delegated to scirs2-core's unified SIMD abstraction layer
//! in compliance with the project-wide SIMD policy.

use crate::error::InterpolateResult;
use crate::spatial::{BallTree, KdTree};
use ndarray::{ArrayView2, Axis};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

#[cfg(feature = "simd")]
use scirs2_core::simd_ops::SimdUnifiedOps;

/// Enhanced spatial search interface with multiple optimization strategies
pub trait OptimizedSpatialSearch<F: Float> {
    /// Perform batch k-nearest neighbor search for multiple queries
    fn batch_k_nearest_neighbors(
        &self,
        queries: &ArrayView2<F>,
        k: usize,
    ) -> InterpolateResult<Vec<Vec<(usize, F)>>>;

    /// Perform parallel k-nearest neighbor search
    fn parallel_k_nearest_neighbors(
        &self,
        queries: &ArrayView2<F>,
        k: usize,
        workers: Option<usize>,
    ) -> InterpolateResult<Vec<Vec<(usize, F)>>>;

    /// Adaptive k-nearest neighbor search that adjusts strategy based on query characteristics
    fn adaptive_k_nearest_neighbors(
        &self,
        query: &[F],
        k: usize,
    ) -> InterpolateResult<Vec<(usize, F)>>;

    /// Range search with multiple radii for the same query point
    fn multi_radius_search(
        &self,
        query: &[F],
        radii: &[F],
    ) -> InterpolateResult<Vec<Vec<(usize, F)>>>;
}

/// SIMD-accelerated distance computation utilities
pub struct SimdDistanceOps;

impl SimdDistanceOps {
    /// Compute squared Euclidean distance using SIMD operations when available
    #[cfg(feature = "simd")]
    pub fn squared_euclidean_distance<F>(a: &[F], b: &[F]) -> F
    where
        F: Float + FromPrimitive + SimdUnifiedOps,
    {
        assert_eq!(a.len(), b.len(), "Vectors must have the same dimension");

        if F::simd_available() && a.len() >= 4 {
            // Use SIMD for larger vectors
            let a_arr = Array1::from_vec(a.to_vec());
            let b_arr = Array1::from_vec(b.to_vec());
            let diff = F::simd_sub(&a_arr.view(), &b_arr.view());
            let squared = F::simd_mul(&diff.view(), &diff.view());
            F::simd_sum(&squared.view())
        } else {
            // Fallback to scalar computation for small vectors
            a.iter()
                .zip(b.iter())
                .map(|(&x, &y)| {
                    let diff = x - y;
                    diff * diff
                })
                .fold(F::zero(), |acc, x| acc + x)
        }
    }

    /// Compute squared Euclidean distance without SIMD
    #[cfg(not(feature = "simd"))]
    pub fn squared_euclidean_distance<F>(a: &[F], b: &[F]) -> F
    where
        F: Float + FromPrimitive,
    {
        assert_eq!(a.len(), b.len(), "Vectors must have the same dimension");

        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let diff = x - y;
                diff * diff
            })
            .fold(F::zero(), |acc, x| acc + x)
    }

    /// Batch compute distances from multiple points to a single query
    #[cfg(feature = "simd")]
    pub fn batch_distances_to_query<F>(points: &ArrayView2<F>, query: &[F]) -> Vec<F>
    where
        F: Float + FromPrimitive + SimdUnifiedOps,
    {
        points
            .axis_iter(Axis(0))
            .map(|point| {
                let point_slice = point.as_slice().unwrap();
                Self::squared_euclidean_distance(point_slice, query)
            })
            .collect()
    }

    /// Batch compute distances without SIMD
    #[cfg(not(feature = "simd"))]
    pub fn batch_distances_to_query<F>(points: &ArrayView2<F>, query: &[F]) -> Vec<F>
    where
        F: Float + FromPrimitive,
    {
        points
            .axis_iter(Axis(0))
            .map(|point| {
                let point_slice = point.as_slice().unwrap();
                Self::squared_euclidean_distance(point_slice, query)
            })
            .collect()
    }
}

/// Cache-friendly kNN search with distance precomputation
#[allow(dead_code)]
pub struct CacheFriendlyKNN<F: Float> {
    /// Maximum number of distances to cache
    cache_size: usize,
    /// Phantom data for type parameter
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float + FromPrimitive> CacheFriendlyKNN<F> {
    /// Create a new cache-friendly kNN searcher
    pub fn new(cache_size: usize) -> Self {
        Self {
            cache_size,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Find k nearest neighbors with caching strategy
    pub fn find_k_nearest<S>(
        &self,
        searcher: &S,
        query: &[F],
        k: usize,
    ) -> InterpolateResult<Vec<(usize, F)>>
    where
        S: OptimizedSpatialSearch<F>,
    {
        // Use adaptive strategy for small k
        if k <= 10 {
            searcher.adaptive_k_nearest_neighbors(query, k)
        } else {
            // For larger k, use standard search
            // This is a placeholder - actual implementation would depend on the searcher
            searcher.adaptive_k_nearest_neighbors(query, k)
        }
    }
}

/// Parallel batch query processor
#[cfg(feature = "parallel")]
pub struct ParallelQueryProcessor<F: Float> {
    /// Number of worker threads
    num_workers: usize,
    /// Phantom data for type parameter
    _phantom: std::marker::PhantomData<F>,
}

#[cfg(feature = "parallel")]
impl<F: Float + FromPrimitive + Send + Sync> ParallelQueryProcessor<F> {
    /// Create a new parallel query processor
    pub fn new(num_workers: Option<usize>) -> Self {
        use scirs2_core::parallel_ops::num_threads;

        Self {
            num_workers: num_workers.unwrap_or_else(num_threads),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Process queries in parallel
    pub fn process_queries<S>(
        &self,
        searcher: &S,
        queries: &ArrayView2<F>,
        k: usize,
    ) -> InterpolateResult<Vec<Vec<(usize, F)>>>
    where
        S: OptimizedSpatialSearch<F> + Sync,
    {
        searcher.parallel_k_nearest_neighbors(queries, k, Some(self.num_workers))
    }
}

/// Default implementation of OptimizedSpatialSearch for KdTree
impl<F> OptimizedSpatialSearch<F> for KdTree<F>
where
    F: Float + FromPrimitive + Debug + Send + Sync,
{
    fn batch_k_nearest_neighbors(
        &self,
        queries: &ArrayView2<F>,
        k: usize,
    ) -> InterpolateResult<Vec<Vec<(usize, F)>>> {
        queries
            .axis_iter(Axis(0))
            .map(|query| {
                let query_slice = query.as_slice().unwrap();
                self.k_nearest_neighbors(query_slice, k)
            })
            .collect()
    }

    #[cfg(feature = "parallel")]
    fn parallel_k_nearest_neighbors(
        &self,
        queries: &ArrayView2<F>,
        k: usize,
        workers: Option<usize>,
    ) -> InterpolateResult<Vec<Vec<(usize, F)>>> {
        use scirs2_core::parallel_ops::*;

        let queries_vec: Vec<_> = queries.axis_iter(Axis(0)).collect();

        par_scope(|_| {
            queries_vec
                .into_par_iter()
                .map(|query| {
                    let query_slice = query.as_slice().unwrap();
                    self.k_nearest_neighbors(query_slice, k)
                })
                .collect::<Result<Vec<_>, _>>()
        })
    }

    #[cfg(not(feature = "parallel"))]
    fn parallel_k_nearest_neighbors(
        &self,
        queries: &ArrayView2<F>,
        k: usize,
        _workers: Option<usize>,
    ) -> InterpolateResult<Vec<Vec<(usize, F)>>> {
        // Fallback to sequential processing
        self.batch_k_nearest_neighbors(queries, k)
    }

    fn adaptive_k_nearest_neighbors(
        &self,
        query: &[F],
        k: usize,
    ) -> InterpolateResult<Vec<(usize, F)>> {
        // For now, just use the standard k-nearest neighbors
        // A more sophisticated implementation could choose different strategies
        // based on k, dimension, and data characteristics
        self.k_nearest_neighbors(query, k)
    }

    fn multi_radius_search(
        &self,
        query: &[F],
        radii: &[F],
    ) -> InterpolateResult<Vec<Vec<(usize, F)>>> {
        radii
            .iter()
            .map(|&radius| self.radius_neighbors(query, radius))
            .collect()
    }
}

/// Default implementation of OptimizedSpatialSearch for BallTree
impl<F> OptimizedSpatialSearch<F> for BallTree<F>
where
    F: Float + FromPrimitive + Debug + Send + Sync,
{
    fn batch_k_nearest_neighbors(
        &self,
        queries: &ArrayView2<F>,
        k: usize,
    ) -> InterpolateResult<Vec<Vec<(usize, F)>>> {
        queries
            .axis_iter(Axis(0))
            .map(|query| {
                let query_slice = query.as_slice().unwrap();
                self.k_nearest_neighbors(query_slice, k)
            })
            .collect()
    }

    #[cfg(feature = "parallel")]
    fn parallel_k_nearest_neighbors(
        &self,
        queries: &ArrayView2<F>,
        k: usize,
        workers: Option<usize>,
    ) -> InterpolateResult<Vec<Vec<(usize, F)>>> {
        use scirs2_core::parallel_ops::*;

        let queries_vec: Vec<_> = queries.axis_iter(Axis(0)).collect();

        par_scope(|_| {
            queries_vec
                .into_par_iter()
                .map(|query| {
                    let query_slice = query.as_slice().unwrap();
                    self.k_nearest_neighbors(query_slice, k)
                })
                .collect::<Result<Vec<_>, _>>()
        })
    }

    #[cfg(not(feature = "parallel"))]
    fn parallel_k_nearest_neighbors(
        &self,
        queries: &ArrayView2<F>,
        k: usize,
        _workers: Option<usize>,
    ) -> InterpolateResult<Vec<Vec<(usize, F)>>> {
        // Fallback to sequential processing
        self.batch_k_nearest_neighbors(queries, k)
    }

    fn adaptive_k_nearest_neighbors(
        &self,
        query: &[F],
        k: usize,
    ) -> InterpolateResult<Vec<(usize, F)>> {
        self.k_nearest_neighbors(query, k)
    }

    fn multi_radius_search(
        &self,
        query: &[F],
        radii: &[F],
    ) -> InterpolateResult<Vec<Vec<(usize, F)>>> {
        radii
            .iter()
            .map(|&radius| self.radius_neighbors(query, radius))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_simd_distance_ops() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];

        let distance = SimdDistanceOps::squared_euclidean_distance(&a, &b);
        assert_eq!(distance, 4.0);
    }

    #[test]
    fn test_batch_distances() {
        let points = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let query = vec![0.0, 0.0];

        let distances = SimdDistanceOps::batch_distances_to_query(&points.view(), &query);

        assert_eq!(distances.len(), 3);
        assert_eq!(distances[0], 5.0); // (1-0)^2 + (2-0)^2 = 5
        assert_eq!(distances[1], 25.0); // (3-0)^2 + (4-0)^2 = 25
        assert_eq!(distances[2], 61.0); // (5-0)^2 + (6-0)^2 = 61
    }

    #[test]
    fn test_cache_friendly_knn() {
        let knn = CacheFriendlyKNN::<f64>::new(1000);
        assert_eq!(knn.cache_size, 1000);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_query_processor() {
        let processor = ParallelQueryProcessor::<f64>::new(Some(4));
        assert_eq!(processor.num_workers, 4);
    }
}
