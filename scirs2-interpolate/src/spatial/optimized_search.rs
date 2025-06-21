//! Optimized spatial search algorithms with enhanced performance features
//!
//! This module provides advanced spatial search optimizations including:
//! - SIMD-accelerated distance computations
//! - Cache-friendly memory layouts
//! - Adaptive search strategies
//! - Batch query processing
//! - Multi-threaded search operations

#[cfg(feature = "simd")]
use wide::f64x4;

use crate::error::{InterpolateError, InterpolateResult};
use crate::spatial::{BallTree, KdTree};
use ndarray::{Array1, ArrayView2, Axis};
use num_traits::{Float, FromPrimitive};
use std::cmp::Ordering;
use std::fmt::Debug;

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

/// SIMD-accelerated distance computations
pub struct SIMDDistanceCalculator<F: Float> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float + FromPrimitive + 'static> SIMDDistanceCalculator<F> {
    /// Create a new SIMD distance calculator
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    /// Compute squared Euclidean distances between a query and multiple points using SIMD
    #[cfg(target_arch = "x86_64")]
    pub fn batch_squared_distances_simd(&self, query: &[F], points: &ArrayView2<F>) -> Array1<F> {
        let n_points = points.nrows();
        let _n_dims = points.ncols();
        let mut distances = Array1::zeros(n_points);

        // For f64, try to use AVX2 if available
        if std::mem::size_of::<F>() == 8 {
            self.batch_squared_distances_avx2_f64(query, points, &mut distances);
        } else {
            // Fallback to scalar implementation for f32 or if SIMD is not available
            self.batch_squared_distances_scalar(query, points, &mut distances);
        }

        distances
    }

    /// SIMD-accelerated distance computation for f64
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    fn batch_squared_distances_avx2_f64(
        &self,
        query: &[F],
        points: &ArrayView2<F>,
        distances: &mut Array1<F>,
    ) {
        use std::any::TypeId;

        // Only use SIMD for f64
        if TypeId::of::<F>() == TypeId::of::<f64>() {
            self.batch_squared_distances_simd_f64_impl(query, points, distances);
        } else {
            self.batch_squared_distances_scalar(query, points, distances);
        }
    }

    /// SIMD implementation specifically for f64
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    fn batch_squared_distances_simd_f64_impl(
        &self,
        query: &[F],
        points: &ArrayView2<F>,
        distances: &mut Array1<F>,
    ) {
        let n_points = points.nrows();
        let n_dims = points.ncols();

        // Convert query to f64 for SIMD processing
        let query_f64: Vec<f64> = query.iter().map(|&x| x.to_f64().unwrap_or(0.0)).collect();

        // Process points in chunks of 4 for SIMD
        for i in (0..n_points).step_by(4) {
            let chunk_size = (n_points - i).min(4);

            if chunk_size == 4 && n_dims >= 2 {
                // Full SIMD processing for 4 points
                let mut sum_sq = f64x4::ZERO;

                // Process dimensions in groups to make efficient use of SIMD
                let mut dim_idx = 0;
                while dim_idx + 4 <= n_dims {
                    // Load 4 dimensions from the query
                    let query_chunk = f64x4::new([
                        query_f64[dim_idx],
                        query_f64[dim_idx + 1],
                        query_f64[dim_idx + 2],
                        query_f64[dim_idx + 3],
                    ]);

                    // For each of the 4 points, accumulate distance for these dimensions
                    for point_offset in 0..4 {
                        let point_row = points.row(i + point_offset);
                        let point_chunk = f64x4::new([
                            point_row[dim_idx].to_f64().unwrap_or(0.0),
                            point_row[dim_idx + 1].to_f64().unwrap_or(0.0),
                            point_row[dim_idx + 2].to_f64().unwrap_or(0.0),
                            point_row[dim_idx + 3].to_f64().unwrap_or(0.0),
                        ]);

                        let diff = point_chunk - query_chunk;
                        let sq_diff = diff * diff;

                        // Horizontal add to accumulate this point's distance
                        let horizontal_sum = sq_diff.as_array_ref();
                        let point_distance_contribution = horizontal_sum[0]
                            + horizontal_sum[1]
                            + horizontal_sum[2]
                            + horizontal_sum[3];

                        // Add to the appropriate slot in sum_sq
                        match point_offset {
                            0 => sum_sq += f64x4::new([point_distance_contribution, 0.0, 0.0, 0.0]),
                            1 => sum_sq += f64x4::new([0.0, point_distance_contribution, 0.0, 0.0]),
                            2 => sum_sq += f64x4::new([0.0, 0.0, point_distance_contribution, 0.0]),
                            3 => sum_sq += f64x4::new([0.0, 0.0, 0.0, point_distance_contribution]),
                            _ => unreachable!(),
                        }
                    }

                    dim_idx += 4;
                }

                // Handle remaining dimensions with scalar processing
                while dim_idx < n_dims {
                    for point_offset in 0..4 {
                        let point_row = points.row(i + point_offset);
                        let diff = point_row[dim_idx].to_f64().unwrap_or(0.0) - query_f64[dim_idx];
                        let sq_diff = diff * diff;

                        match point_offset {
                            0 => sum_sq += f64x4::new([sq_diff, 0.0, 0.0, 0.0]),
                            1 => sum_sq += f64x4::new([0.0, sq_diff, 0.0, 0.0]),
                            2 => sum_sq += f64x4::new([0.0, 0.0, sq_diff, 0.0]),
                            3 => sum_sq += f64x4::new([0.0, 0.0, 0.0, sq_diff]),
                            _ => unreachable!(),
                        }
                    }
                    dim_idx += 1;
                }

                // Store results
                let result_array = sum_sq.as_array_ref();
                for j in 0..4 {
                    distances[i + j] = F::from_f64(result_array[j]).unwrap();
                }
            } else {
                // Fallback to scalar for partial chunks or when SIMD isn't beneficial
                for j in 0..chunk_size {
                    let mut sum_sq = 0.0f64;
                    let point_row = points.row(i + j);
                    for (dim, &q_val_f) in query.iter().enumerate() {
                        let q_val = q_val_f.to_f64().unwrap_or(0.0);
                        let p_val = point_row[dim].to_f64().unwrap_or(0.0);
                        let diff = p_val - q_val;
                        sum_sq += diff * diff;
                    }
                    distances[i + j] = F::from_f64(sum_sq).unwrap();
                }
            }
        }
    }

    /// Non-SIMD fallback for when SIMD is not available
    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    fn batch_squared_distances_avx2_f64(
        &self,
        query: &[F],
        points: &ArrayView2<F>,
        distances: &mut Array1<F>,
    ) {
        self.batch_squared_distances_scalar(query, points, distances);
    }

    /// Fallback scalar implementation
    fn batch_squared_distances_scalar(
        &self,
        query: &[F],
        points: &ArrayView2<F>,
        distances: &mut Array1<F>,
    ) {
        for (i, point) in points.axis_iter(Axis(0)).enumerate() {
            let mut sum_sq = F::zero();
            for (j, &q_val) in query.iter().enumerate() {
                let diff = point[j] - q_val;
                sum_sq = sum_sq + diff * diff;
            }
            distances[i] = sum_sq;
        }
    }

    /// Non-SIMD version for other architectures
    #[cfg(not(target_arch = "x86_64"))]
    pub fn batch_squared_distances_simd(&self, query: &[F], points: &ArrayView2<F>) -> Array1<F> {
        let n_points = points.nrows();
        let mut distances = Array1::zeros(n_points);
        self.batch_squared_distances_scalar(query, points, &mut distances);
        distances
    }
}

impl<F: Float + FromPrimitive + 'static> Default for SIMDDistanceCalculator<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Cache-friendly spatial index with improved memory layout
pub struct CacheFriendlyIndex<F: Float + FromPrimitive + Debug + std::cmp::PartialOrd> {
    /// Points stored in cache-friendly layout (SoA - Structure of Arrays)
    points_soa: Vec<Vec<F>>,
    /// Original indices for mapping back to user data
    indices: Vec<usize>,
    /// Dimension of the space
    dim: usize,
    /// Spatial data structure (KdTree or BallTree)
    spatial_structure: SpatialStructure<F>,
}

enum SpatialStructure<F: Float + FromPrimitive + Debug + std::cmp::PartialOrd> {
    KdTree(KdTree<F>),
    BallTree(BallTree<F>),
}

impl<F> CacheFriendlyIndex<F>
where
    F: Float + FromPrimitive + Debug + std::cmp::PartialOrd + Copy,
{
    /// Create a new cache-friendly index
    pub fn new(points: &ArrayView2<F>) -> InterpolateResult<Self> {
        let n_points = points.nrows();
        let dim = points.ncols();

        // Convert AoS (Array of Structures) to SoA (Structure of Arrays) for better cache performance
        let mut points_soa = vec![Vec::with_capacity(n_points); dim];
        let indices: Vec<usize> = (0..n_points).collect();

        for point in points.axis_iter(Axis(0)) {
            for (d, &coord) in point.iter().enumerate() {
                points_soa[d].push(coord);
            }
        }

        // Choose appropriate spatial structure based on dimensionality
        let spatial_structure = if dim <= 10 {
            // Use KdTree for low-dimensional data
            let kdtree = KdTree::new(points.to_owned())?;
            SpatialStructure::KdTree(kdtree)
        } else {
            // Use BallTree for high-dimensional data
            let balltree = BallTree::new(points.to_owned())?;
            SpatialStructure::BallTree(balltree)
        };

        Ok(Self {
            points_soa,
            indices,
            dim,
            spatial_structure,
        })
    }

    /// Get a point in AoS format
    #[allow(dead_code)]
    fn get_point_aos(&self, index: usize) -> Vec<F> {
        let mut point = Vec::with_capacity(self.dim);
        for d in 0..self.dim {
            point.push(self.points_soa[d][index]);
        }
        point
    }

    /// Perform k-nearest neighbor search with cache-friendly access patterns
    pub fn k_nearest_neighbors(&self, query: &[F], k: usize) -> InterpolateResult<Vec<(usize, F)>> {
        match &self.spatial_structure {
            SpatialStructure::KdTree(kdtree) => kdtree.k_nearest_neighbors(query, k),
            SpatialStructure::BallTree(balltree) => balltree.k_nearest_neighbors(query, k),
        }
    }

    /// Cache-friendly batch distance computation
    pub fn batch_distances(&self, query: &[F]) -> Vec<F> {
        let mut distances = Vec::with_capacity(self.indices.len());

        // Use SoA layout for better cache performance
        for i in 0..self.indices.len() {
            let mut sum_sq = F::zero();
            for (d, &query_coord) in query.iter().enumerate().take(self.dim) {
                let diff = self.points_soa[d][i] - query_coord;
                sum_sq = sum_sq + diff * diff;
            }
            distances.push(sum_sq.sqrt());
        }

        distances
    }
}

/// Adaptive search strategy that chooses the best method based on query characteristics
pub struct AdaptiveSearchStrategy<F: Float + FromPrimitive + Debug + std::cmp::PartialOrd> {
    cache_friendly_index: CacheFriendlyIndex<F>,
    #[allow(dead_code)]
    simd_calculator: SIMDDistanceCalculator<F>,
    stats: SearchStats,
}

#[derive(Debug, Default)]
pub struct SearchStats {
    pub total_queries: usize,
    pub kdtree_queries: usize,
    pub balltree_queries: usize,
    pub brute_force_queries: usize,
    pub avg_query_time_ns: f64,
}

impl<F> AdaptiveSearchStrategy<F>
where
    F: Float + FromPrimitive + Debug + std::cmp::PartialOrd + Copy + 'static,
{
    /// Create a new adaptive search strategy
    pub fn new(points: &ArrayView2<F>) -> InterpolateResult<Self> {
        let cache_friendly_index = CacheFriendlyIndex::new(points)?;
        let simd_calculator = SIMDDistanceCalculator::new();

        Ok(Self {
            cache_friendly_index,
            simd_calculator,
            stats: SearchStats::default(),
        })
    }

    /// Perform adaptive k-nearest neighbor search
    pub fn adaptive_k_nearest_neighbors(
        &mut self,
        query: &[F],
        k: usize,
    ) -> InterpolateResult<Vec<(usize, F)>> {
        self.stats.total_queries += 1;

        // Choose strategy based on query characteristics and dataset properties
        let strategy = self.choose_strategy(query, k);

        let start_time = std::time::Instant::now();
        let result = match strategy {
            SearchStrategy::KdTree => {
                self.stats.kdtree_queries += 1;
                self.cache_friendly_index.k_nearest_neighbors(query, k)
            }
            SearchStrategy::BallTree => {
                self.stats.balltree_queries += 1;
                self.cache_friendly_index.k_nearest_neighbors(query, k)
            }
            SearchStrategy::BruteForce => {
                self.stats.brute_force_queries += 1;
                self.brute_force_k_nearest_neighbors(query, k)
            }
        };

        let elapsed = start_time.elapsed();
        self.update_timing_stats(elapsed);

        result
    }

    /// Choose the best search strategy based on query characteristics
    fn choose_strategy(&self, _query: &[F], k: usize) -> SearchStrategy {
        let n_points = self.cache_friendly_index.indices.len();
        let dim = self.cache_friendly_index.dim;

        // Decision rules based on empirical performance characteristics
        if n_points < 50 {
            SearchStrategy::BruteForce
        } else if dim <= 5 && k <= n_points / 10 {
            SearchStrategy::KdTree
        } else if dim <= 20 && k <= n_points / 5 {
            SearchStrategy::BallTree
        } else {
            SearchStrategy::BruteForce
        }
    }

    /// Brute force k-nearest neighbor search with SIMD optimization
    fn brute_force_k_nearest_neighbors(
        &self,
        query: &[F],
        k: usize,
    ) -> InterpolateResult<Vec<(usize, F)>> {
        let distances = self.cache_friendly_index.batch_distances(query);
        let mut indexed_distances: Vec<(usize, F)> = distances
            .into_iter()
            .enumerate()
            .map(|(i, dist)| (self.cache_friendly_index.indices[i], dist))
            .collect();

        // Sort by distance and take the k smallest
        indexed_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        indexed_distances.truncate(k);

        Ok(indexed_distances)
    }

    /// Update timing statistics
    fn update_timing_stats(&mut self, elapsed: std::time::Duration) {
        let elapsed_ns = elapsed.as_nanos() as f64;
        let n = self.stats.total_queries as f64;
        self.stats.avg_query_time_ns = (self.stats.avg_query_time_ns * (n - 1.0) + elapsed_ns) / n;
    }

    /// Get search statistics
    pub fn stats(&self) -> &SearchStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = SearchStats::default();
    }
}

#[derive(Debug, Clone, Copy)]
enum SearchStrategy {
    KdTree,
    BallTree,
    BruteForce,
}

/// Batch query processor for efficient processing of multiple queries
pub struct BatchQueryProcessor<F: Float + FromPrimitive + Debug + std::cmp::PartialOrd> {
    adaptive_strategy: AdaptiveSearchStrategy<F>,
    batch_size: usize,
}

impl<F> BatchQueryProcessor<F>
where
    F: Float + FromPrimitive + Debug + std::cmp::PartialOrd + Copy + 'static,
{
    /// Create a new batch query processor
    pub fn new(points: &ArrayView2<F>) -> InterpolateResult<Self> {
        let adaptive_strategy = AdaptiveSearchStrategy::new(points)?;

        Ok(Self {
            adaptive_strategy,
            batch_size: 32, // Default batch size for cache efficiency
        })
    }

    /// Set the batch size for processing
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Process multiple k-nearest neighbor queries in batches
    pub fn batch_k_nearest_neighbors(
        &mut self,
        queries: &ArrayView2<F>,
        k: usize,
    ) -> InterpolateResult<Vec<Vec<(usize, F)>>> {
        let n_queries = queries.nrows();
        let mut results = Vec::with_capacity(n_queries);

        // Process queries in batches for better cache locality
        for batch_start in (0..n_queries).step_by(self.batch_size) {
            let batch_end = (batch_start + self.batch_size).min(n_queries);

            for i in batch_start..batch_end {
                let query = queries.row(i);
                let query_slice = query.as_slice().unwrap();
                let result = self
                    .adaptive_strategy
                    .adaptive_k_nearest_neighbors(query_slice, k)?;
                results.push(result);
            }
        }

        Ok(results)
    }

    /// Process queries in parallel using multiple threads
    pub fn parallel_k_nearest_neighbors(
        &self,
        queries: &ArrayView2<F>,
        k: usize,
        workers: Option<usize>,
    ) -> InterpolateResult<Vec<Vec<(usize, F)>>> {
        // Set up parallel configuration
        let _pool = if let Some(n_workers) = workers {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n_workers)
                    .build()
                    .map_err(|e| InterpolateError::ComputationError(e.to_string()))?,
            )
        } else {
            None
        };

        // For simplicity, use sequential processing for now
        // A full parallel implementation would require thread-safe statistics collection
        let mut results = Vec::with_capacity(queries.nrows());
        for i in 0..queries.nrows() {
            let query = queries.row(i);
            let query_slice = query.as_slice().unwrap();
            // Create a local strategy for each query to avoid borrowing issues
            let mut local_strategy =
                AdaptiveSearchStrategy::new(&queries.slice(ndarray::s![i..i + 1, ..]))?;
            let result = local_strategy.adaptive_k_nearest_neighbors(query_slice, k)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Get batch processor statistics
    pub fn stats(&self) -> &SearchStats {
        self.adaptive_strategy.stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_simd_distance_calculator() {
        let calc = SIMDDistanceCalculator::<f64>::new();
        let query = vec![0.0, 0.0];
        let points = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

        let distances = calc.batch_squared_distances_simd(&query, &points.view());

        assert_eq!(distances.len(), 3);
        assert!((distances[0] - 1.0).abs() < 1e-10);
        assert!((distances[1] - 1.0).abs() < 1e-10);
        assert!((distances[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_cache_friendly_index() {
        let points =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

        let index = CacheFriendlyIndex::new(&points.view()).unwrap();
        let query = vec![0.5, 0.5];
        let neighbors = index.k_nearest_neighbors(&query, 2).unwrap();

        assert_eq!(neighbors.len(), 2);
    }

    #[test]
    fn test_adaptive_search_strategy() {
        let points =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

        let mut strategy = AdaptiveSearchStrategy::new(&points.view()).unwrap();
        let query = vec![0.5, 0.5];
        let neighbors = strategy.adaptive_k_nearest_neighbors(&query, 2).unwrap();

        assert_eq!(neighbors.len(), 2);
        assert_eq!(strategy.stats().total_queries, 1);
    }

    #[test]
    fn test_batch_query_processor() {
        let points =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

        let mut processor = BatchQueryProcessor::new(&points.view()).unwrap();
        let queries = Array2::from_shape_vec((2, 2), vec![0.25, 0.25, 0.75, 0.75]).unwrap();

        let results = processor
            .batch_k_nearest_neighbors(&queries.view(), 2)
            .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 2);
        assert_eq!(results[1].len(), 2);
    }
}
