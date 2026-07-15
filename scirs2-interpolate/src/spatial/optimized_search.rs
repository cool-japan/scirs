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

use crate::error::{InterpolateError, InterpolateResult};
use crate::spatial::{BallTree, KdTree};
use scirs2_core::ndarray::{ArrayView2, Axis};

#[cfg(feature = "simd")]
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::numeric::{Float, FromPrimitive};
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

        if F::simd_available() {
            F::simd_distance_squared_euclidean(&ArrayView1::from(a), &ArrayView1::from(b))
        } else {
            a.iter()
                .zip(b.iter())
                .map(|(&x, &y)| {
                    let diff = x - y;
                    diff * diff
                })
                .fold(F::zero(), |acc, x| acc + x)
        }
    }

    /// Enhanced batch distance computation with SIMD optimization for better memory access patterns
    #[cfg(feature = "simd")]
    pub fn enhanced_batch_distances<F>(
        points: &ArrayView2<F>,
        queries: &ArrayView2<F>,
    ) -> Vec<Vec<F>>
    where
        F: Float + FromPrimitive + SimdUnifiedOps + Debug,
    {
        let n_queries = queries.nrows();
        let n_points = points.nrows();
        let dim = points.ncols();

        let mut results = Vec::with_capacity(n_queries);

        for query_idx in 0..n_queries {
            let query = queries.row(query_idx);
            let mut distances = Vec::with_capacity(n_points);

            if F::simd_available() && dim >= 4 && n_points >= 8 {
                // Process in chunks for better cache utilization
                const CHUNK_SIZE: usize = 16;

                for chunk_start in (0..n_points).step_by(CHUNK_SIZE) {
                    let chunk_end = (chunk_start + CHUNK_SIZE).min(n_points);

                    for point_idx in chunk_start..chunk_end {
                        let point = points.row(point_idx);

                        // Use SIMD-optimized distance calculation
                        let distance = if dim >= 8 {
                            // For higher dimensions, use vectorized operations
                            let diff = F::simd_sub(&point, &query);
                            let squared = F::simd_mul(&diff.view(), &diff.view());
                            F::simd_sum(&squared.view())
                        } else {
                            // Fallback for lower dimensions
                            Self::squared_euclidean_distance(
                                point.as_slice().expect("Operation failed"),
                                query.as_slice().expect("Operation failed"),
                            )
                        };

                        distances.push(distance);
                    }
                }
            } else {
                // Non-SIMD fallback
                for point_idx in 0..n_points {
                    let point = points.row(point_idx);
                    let distance = Self::squared_euclidean_distance(
                        point.as_slice().expect("Operation failed"),
                        query.as_slice().expect("Operation failed"),
                    );
                    distances.push(distance);
                }
            }

            results.push(distances);
        }

        results
    }

    /// SIMD-optimized parallel batch processing for very large datasets
    #[cfg(all(feature = "simd", feature = "parallel"))]
    pub fn parallel_enhanced_batch_distances<F>(
        points: &ArrayView2<F>,
        queries: &ArrayView2<F>,
        _num_threads: Option<usize>,
    ) -> Vec<Vec<F>>
    where
        F: Float + FromPrimitive + SimdUnifiedOps + Debug + Send + Sync,
    {
        let n_queries = queries.nrows();

        // Process queries sequentially for now
        (0..n_queries)
            .map(|query_idx| {
                let query = queries.row(query_idx);
                Self::batch_distances_to_query(points, query.as_slice().expect("Operation failed"))
            })
            .collect()
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
                let point_slice = point.as_slice().expect("Operation failed");
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
                let point_slice = point.as_slice().expect("Operation failed");
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
    pub fn new(cachesize: usize) -> Self {
        Self {
            cache_size: cachesize,
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
    F: Float + FromPrimitive + Debug + Send + Sync + ordered_float::FloatCore,
{
    fn batch_k_nearest_neighbors(
        &self,
        queries: &ArrayView2<F>,
        k: usize,
    ) -> InterpolateResult<Vec<Vec<(usize, F)>>> {
        queries
            .axis_iter(Axis(0))
            .map(|query| {
                let query_slice = query.as_slice().expect("Operation failed");
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
                    let query_slice = query.as_slice().expect("Operation failed");
                    self.k_nearest_neighbors(query_slice, k)
                })
                .collect::<Result<Vec<_>, InterpolateError>>()
        })
    }

    #[cfg(not(feature = "parallel"))]
    fn parallel_k_nearest_neighbors(
        &self,
        queries: &ArrayView2<F>,
        k: usize,
        workers: Option<usize>,
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
    F: Float + FromPrimitive + Debug + Send + Sync + ordered_float::FloatCore,
{
    fn batch_k_nearest_neighbors(
        &self,
        queries: &ArrayView2<F>,
        k: usize,
    ) -> InterpolateResult<Vec<Vec<(usize, F)>>> {
        queries
            .axis_iter(Axis(0))
            .map(|query| {
                let query_slice = query.as_slice().expect("Operation failed");
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
                    let query_slice = query.as_slice().expect("Operation failed");
                    self.k_nearest_neighbors(query_slice, k)
                })
                .collect::<Result<Vec<_>, InterpolateError>>()
        })
    }

    #[cfg(not(feature = "parallel"))]
    fn parallel_k_nearest_neighbors(
        &self,
        queries: &ArrayView2<F>,
        k: usize,
        workers: Option<usize>,
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
    use scirs2_core::ndarray::array;

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

    /// Recursively invokes `SimdDistanceOps::squared_euclidean_distance` (the SIMD-enabled
    /// path) at real Rust call-stack recursion depth (not a loop). Exists purely to
    /// stress-test the `#[inline(never)]` mitigation applied to the underlying SIMD leaf
    /// kernels in scirs2-core (`simd/distances.rs`): the originally hypothesized failure
    /// mode was that a deeply recursive caller (e.g. KdTree/BallTree descent) duplicates
    /// the kernels' wide `__m256`/`__m256d` stack frames at every recursion level.
    #[cfg(feature = "simd")]
    fn recursive_squared_distance_probe<F>(depth: usize, a: &[F], b: &[F], acc: F) -> F
    where
        F: Float + FromPrimitive + SimdUnifiedOps,
    {
        let d = SimdDistanceOps::squared_euclidean_distance(a, b);
        if depth == 0 {
            acc + d
        } else {
            recursive_squared_distance_probe(depth - 1, a, b, acc + d)
        }
    }

    /// STRESS TEST: deep, real recursion (not a token 3-level test) calling the
    /// SIMD-enabled `squared_euclidean_distance` at every level, run inside a thread with
    /// an explicit, bounded stack. Positively confirms no stack overflow occurs with SIMD
    /// enabled — this is the regression guard for the `#[inline(never)]` precautionary
    /// mitigation on `simd_distance_squared_euclidean_f32/f64`.
    #[cfg(feature = "simd")]
    #[test]
    fn test_squared_euclidean_distance_deep_recursion_stress() {
        const DEPTH: usize = 100_000;
        const DIM: usize = 64;
        const STACK_SIZE: usize = 64 * 1024 * 1024; // 64 MiB: explicit, deterministic budget

        // f64 path
        let a64: Vec<f64> = (0..DIM).map(|i| i as f64).collect();
        let b64: Vec<f64> = (0..DIM).map(|i| i as f64 + 1.0).collect();
        let handle64 = std::thread::Builder::new()
            .name("sq-euclid-recursion-stress-f64".to_string())
            .stack_size(STACK_SIZE)
            .spawn(move || recursive_squared_distance_probe(DEPTH, &a64, &b64, 0.0f64))
            .expect("failed to spawn f64 stress-test thread");
        let total64 = handle64.join().expect(
            "deep recursive squared_euclidean_distance (f64, SIMD-enabled) overflowed the stack",
        );
        let expected64 = (DEPTH as f64 + 1.0) * (DIM as f64);
        assert!(
            (total64 - expected64).abs() < 1e-6,
            "f64 stress result mismatch: got {total64}, expected {expected64}"
        );

        // f32 path
        let a32: Vec<f32> = (0..DIM).map(|i| i as f32).collect();
        let b32: Vec<f32> = (0..DIM).map(|i| i as f32 + 1.0).collect();
        let handle32 = std::thread::Builder::new()
            .name("sq-euclid-recursion-stress-f32".to_string())
            .stack_size(STACK_SIZE)
            .spawn(move || recursive_squared_distance_probe(DEPTH, &a32, &b32, 0.0f32))
            .expect("failed to spawn f32 stress-test thread");
        let total32 = handle32.join().expect(
            "deep recursive squared_euclidean_distance (f32, SIMD-enabled) overflowed the stack",
        );
        let expected32 = (DEPTH as f32 + 1.0) * (DIM as f32);
        assert!(
            (total32 - expected32).abs() < 1e-3,
            "f32 stress result mismatch: got {total32}, expected {expected32}"
        );
    }

    /// STRESS TEST: large-scale (>=10,000 points), realistic-depth KdTree/BallTree
    /// build+query combined with direct `SimdDistanceOps` batch calls, complementing the
    /// deep-recursion test above with a large, real-world-shaped workload.
    ///
    /// NOTE (KdTree correctness, out of scope here): while developing this test,
    /// `KdTree::k_nearest_neighbors` was found to return a non-minimal nearest-neighbor
    /// distance at this scale. Root cause (confirmed by inspection of
    /// `spatial/kdtree.rs::build_subtree`, the `n_points <= self.leaf_size` branch): a
    /// leaf node stores only `indices[0]` — the other up to `leaf_size - 1` points in
    /// that partition are never inserted into the tree and can never be returned by any
    /// query. Every pre-existing KdTree test uses <= 5 points (below the default
    /// `leaf_size` of 10), so all of them take the `linear_k_nearest_neighbors` fallback
    /// and never exercise `build_subtree`'s recursive path, which is presumably why this
    /// has gone uncaught. This is a real, separate correctness bug outside this SIMD-
    /// surfacing item's file list (`kdtree.rs` is not touched here) and is flagged for a
    /// dedicated follow-up rather than fixed inline. `BallTree` does not share this bug —
    /// its leaf nodes retain all member indices (`BallNode.indices: Vec<usize>`) and its
    /// `search_k_nearest` iterates all of them — so only `BallTree`'s answer is
    /// cross-checked against the brute-force SIMD minimum below. `KdTree` is still built
    /// and queried here to confirm it does not crash/overflow the stack at this scale,
    /// which is what this test is actually chartered to prove.
    #[cfg(feature = "simd")]
    #[test]
    fn test_squared_euclidean_distance_large_kdtree_balltree_stress() {
        use scirs2_core::ndarray::Array2;

        const N_POINTS: usize = 12_000;
        const DIM: usize = 8;

        // Deterministic LCG-based point generation (matches the project's established
        // reproducible-PRNG idiom elsewhere in this crate; avoids a `rand` dev-dependency).
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next_f64 = || -> f64 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };

        let points = Array2::from_shape_fn((N_POINTS, DIM), |_| next_f64() * 100.0);
        let query_points = Array2::from_shape_fn((32, DIM), |_| next_f64() * 100.0);

        // Build both tree types at realistic depth (~log2(12_000) ~= 14 levels).
        let kdtree = KdTree::new(points.clone()).expect("KdTree build should succeed");
        let balltree = BallTree::new(points.clone()).expect("BallTree build should succeed");

        for query in query_points.axis_iter(Axis(0)) {
            let query_slice = query.as_slice().expect("contiguous query row");

            // Exercise the real recursive tree descent at realistic depth for BOTH trees
            // (this "does it crash/overflow" check is what this test is chartered to
            // prove; see the KdTree correctness note on the test above for why only
            // BallTree's *answer* is cross-checked below).
            let kd_neighbors = kdtree
                .k_nearest_neighbors(query_slice, 10)
                .expect("KdTree k-NN should succeed");
            let ball_neighbors = balltree
                .k_nearest_neighbors(query_slice, 10)
                .expect("BallTree k-NN should succeed");
            assert_eq!(kd_neighbors.len(), 10);
            assert_eq!(ball_neighbors.len(), 10);

            // Exercise SimdDistanceOps::squared_euclidean_distance directly against every
            // point at this scale (batch_distances_to_query delegates to it per-row).
            let distances = SimdDistanceOps::batch_distances_to_query(&points.view(), query_slice);
            assert_eq!(distances.len(), N_POINTS);

            // Cross-check: BallTree's best (sqrt'd Euclidean) neighbor distance, squared,
            // should match the minimum of the directly SIMD-computed squared distances.
            let min_direct = distances.iter().cloned().fold(f64::INFINITY, f64::min);
            let ball_best_dist_sq = ball_neighbors[0].1.powi(2);
            assert!(
                (ball_best_dist_sq - min_direct).abs() < 1e-6,
                "BallTree best squared dist {ball_best_dist_sq} should match direct SIMD min {min_direct}"
            );
        }
    }
}
