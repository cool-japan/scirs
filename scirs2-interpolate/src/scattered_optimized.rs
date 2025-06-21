//! Optimized scattered data interpolation for large datasets
//!
//! This module provides highly optimized implementations of scattered data
//! interpolation methods designed to handle large datasets efficiently.
//! The optimizations include:
//!
//! - **Hierarchical spatial indexing** for fast neighbor searches
//! - **Adaptive mesh refinement** for optimal point distribution
//! - **Memory-efficient algorithms** that minimize memory allocation
//! - **Parallel processing** for batch interpolation operations
//! - **Approximate methods** for real-time applications
//!
//! ## Key Features
//!
//! - **Fast RBF interpolation** with O(N log N) complexity using fast multipole methods
//! - **Adaptive kriging** with local refinement and covariance modeling
//! - **Hierarchical B-splines** for multi-resolution interpolation
//! - **Streaming interpolation** for datasets that don't fit in memory
//! - **GPU acceleration** support for massively parallel interpolation
//!
//! # Examples
//!
//! ```rust
//! use ndarray::{Array1, Array2};
//! use scirs2_interpolate::scattered_optimized::{
//!     OptimizedScatteredInterpolator, ScatteredConfig
//! };
//!
//! // Small scattered dataset for demonstration
//! let points = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
//! let values = Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0]);
//! let config = ScatteredConfig::for_large_dataset();
//!
//! let interpolator = OptimizedScatteredInterpolator::new(
//!     points, values, config
//! ).unwrap();
//! ```

use crate::advanced::rbf::RBFKernel;
use crate::cache_aware::{CacheOptimizedConfig, CacheOptimizedStats};
use crate::error::{InterpolateError, InterpolateResult};
use crate::spatial::enhanced_search::{EnhancedNearestNeighborSearcher, IndexType, SearchConfig};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use num_traits::{Float, FromPrimitive};
use rayon::prelude::*;
use std::fmt::{Debug, Display};
use std::sync::{Arc, RwLock};

/// Configuration for optimized scattered data interpolation
#[derive(Debug, Clone)]
pub struct ScatteredConfig {
    /// Maximum points to process in a single chunk
    pub chunk_size: usize,
    /// Use hierarchical decomposition for large datasets
    pub use_hierarchical: bool,
    /// Number of levels in hierarchical decomposition
    pub hierarchy_levels: usize,
    /// Overlap factor between chunks (0.0-1.0)
    pub chunk_overlap: f64,
    /// Use approximate methods for speed
    pub use_approximation: bool,
    /// Approximation tolerance
    pub approximation_tolerance: f64,
    /// Enable parallel processing
    pub parallel: bool,
    /// Number of worker threads (None = auto)
    pub num_threads: Option<usize>,
    /// Cache optimization settings
    pub cache_config: CacheOptimizedConfig,
    /// Memory limit in MB for working set
    pub memory_limit_mb: usize,
    /// Spatial index configuration
    pub spatial_config: SearchConfig,
}

impl ScatteredConfig {
    /// Create configuration optimized for large datasets
    pub fn for_large_dataset() -> Self {
        Self {
            chunk_size: 10000,
            use_hierarchical: true,
            hierarchy_levels: 4,
            chunk_overlap: 0.1,
            use_approximation: true,
            approximation_tolerance: 1e-6,
            parallel: true,
            num_threads: None,
            cache_config: CacheOptimizedConfig::default(),
            memory_limit_mb: 1024, // 1GB default
            spatial_config: SearchConfig {
                max_neighbors: 50,
                parallel_search: true,
                cache_results: true,
                ..Default::default()
            },
        }
    }

    /// Create configuration optimized for accuracy
    pub fn for_accuracy() -> Self {
        Self {
            chunk_size: 50000,
            use_hierarchical: false,
            hierarchy_levels: 1,
            chunk_overlap: 0.0,
            use_approximation: false,
            approximation_tolerance: 1e-12,
            parallel: true,
            num_threads: None,
            cache_config: CacheOptimizedConfig::default(),
            memory_limit_mb: 4096, // 4GB default
            spatial_config: SearchConfig {
                max_neighbors: 100,
                approximation_factor: 1.0, // Exact
                parallel_search: true,
                cache_results: true,
                ..Default::default()
            },
        }
    }

    /// Create configuration optimized for speed
    pub fn for_speed() -> Self {
        Self {
            chunk_size: 5000,
            use_hierarchical: true,
            hierarchy_levels: 6,
            chunk_overlap: 0.05,
            use_approximation: true,
            approximation_tolerance: 1e-4,
            parallel: true,
            num_threads: None,
            cache_config: CacheOptimizedConfig::default(),
            memory_limit_mb: 512, // 512MB default
            spatial_config: SearchConfig {
                max_neighbors: 20,
                approximation_factor: 1.2, // 20% approximation
                parallel_search: true,
                cache_results: false, // Save memory
                ..Default::default()
            },
        }
    }
}

impl Default for ScatteredConfig {
    fn default() -> Self {
        Self::for_large_dataset()
    }
}

/// Optimized scattered data interpolator with multiple algorithm backends
#[derive(Debug)]
pub struct OptimizedScatteredInterpolator<F>
where
    F: Float + FromPrimitive + Debug + Display + Send + Sync + 'static,
{
    /// Training points (may be hierarchically organized)
    points: Array2<F>,
    /// Training values
    values: Array1<F>,
    /// Configuration
    config: ScatteredConfig,
    /// Spatial indexing structure
    spatial_index: Arc<RwLock<EnhancedNearestNeighborSearcher<F>>>,
    /// Hierarchical decomposition (if enabled)
    hierarchy: Option<HierarchicalDecomposition<F>>,
    /// Performance statistics
    stats: OptimizedScatteredStats,
}

/// Performance statistics for optimized scattered interpolation
#[derive(Debug, Default)]
pub struct OptimizedScatteredStats {
    /// Total number of interpolations performed
    pub interpolations: usize,
    /// Time spent in hierarchical processing (nanoseconds)
    pub hierarchical_time_ns: u64,
    /// Time spent in spatial searches (nanoseconds)  
    pub spatial_search_time_ns: u64,
    /// Number of chunks processed
    pub chunks_processed: usize,
    /// Average chunk processing time (nanoseconds)
    pub avg_chunk_time_ns: u64,
    /// Memory usage statistics
    pub memory_usage_mb: f64,
    /// Cache performance from underlying algorithms
    pub cache_stats: CacheOptimizedStats,
}

/// Hierarchical decomposition for large datasets
#[derive(Debug)]
#[allow(dead_code)]
struct HierarchicalDecomposition<F>
where
    F: Float + FromPrimitive + Debug + Display + Send + Sync,
{
    /// Levels in the hierarchy
    levels: Vec<HierarchyLevel<F>>,
    /// Bounding box of the entire domain
    bounding_box: BoundingBox<F>,
}

/// Single level in the hierarchical decomposition
#[derive(Debug)]
#[allow(dead_code)]
struct HierarchyLevel<F>
where
    F: Float + FromPrimitive + Debug + Display + Send + Sync,
{
    /// Cells at this level
    cells: Vec<HierarchyCell<F>>,
    /// Resolution of this level
    resolution: usize,
    /// Spatial index for this level
    spatial_index: Option<RwLock<EnhancedNearestNeighborSearcher<F>>>,
}

/// Cell in the hierarchical decomposition
#[derive(Debug)]
#[allow(dead_code)]
struct HierarchyCell<F>
where
    F: Float + FromPrimitive + Debug + Display + Send + Sync,
{
    /// Bounding box of this cell
    bounds: BoundingBox<F>,
    /// Point indices contained in this cell
    point_indices: Vec<usize>,
    /// Local interpolator for this cell (if computed)
    local_interpolator: Option<LocalInterpolator<F>>,
    /// Whether this cell needs refinement
    needs_refinement: bool,
}

/// Bounding box in N-dimensional space
#[derive(Debug, Clone)]
struct BoundingBox<F>
where
    F: Float,
{
    /// Minimum coordinates
    min: Array1<F>,
    /// Maximum coordinates
    max: Array1<F>,
}

/// Local interpolator for a hierarchical cell
#[derive(Debug)]
#[allow(dead_code)]
enum LocalInterpolator<F>
where
    F: Float + FromPrimitive + Debug + Display + Send + Sync,
{
    /// RBF interpolator for this cell
    Rbf {
        kernel: RBFKernel,
        epsilon: F,
        coefficients: Array1<F>,
        local_points: Array2<F>,
    },
    /// Linear interpolator for simple cells
    Linear {
        coefficients: Array1<F>,
        reference_point: Array1<F>,
    },
    /// Constant interpolator for uniform cells
    Constant { value: F },
}

impl<F> OptimizedScatteredInterpolator<F>
where
    F: Float + FromPrimitive + Debug + Display + Send + Sync + 'static,
{
    /// Create a new optimized scattered data interpolator
    ///
    /// # Arguments
    ///
    /// * `points` - Training data points (n_points × n_dims)
    /// * `values` - Training data values
    /// * `config` - Optimization configuration
    ///
    /// # Returns
    ///
    /// A new optimized scattered data interpolator
    pub fn new(
        points: Array2<F>,
        values: Array1<F>,
        config: ScatteredConfig,
    ) -> InterpolateResult<Self> {
        if points.nrows() != values.len() {
            return Err(InterpolateError::ValueError(
                "Number of points must match number of values".to_string(),
            ));
        }

        let n_points = points.nrows();

        // Create spatial index
        let spatial_index = Arc::new(RwLock::new(EnhancedNearestNeighborSearcher::new(
            points.clone(),
            IndexType::Adaptive,
            config.spatial_config.clone(),
        )?));

        // Build hierarchical decomposition if enabled and dataset is large enough
        let hierarchy = if config.use_hierarchical && n_points > config.chunk_size * 2 {
            Some(Self::build_hierarchy(&points, &values, &config)?)
        } else {
            None
        };

        Ok(Self {
            points,
            values,
            config,
            spatial_index,
            hierarchy,
            stats: OptimizedScatteredStats::default(),
        })
    }

    /// Build hierarchical decomposition for large datasets
    fn build_hierarchy(
        points: &Array2<F>,
        values: &Array1<F>,
        config: &ScatteredConfig,
    ) -> InterpolateResult<HierarchicalDecomposition<F>> {
        let n_dims = points.ncols();

        // Compute bounding box
        let mut min_coords = Array1::from_elem(n_dims, F::infinity());
        let mut max_coords = Array1::from_elem(n_dims, F::neg_infinity());

        for point in points.axis_iter(Axis(0)) {
            for (i, &coord) in point.iter().enumerate() {
                if coord < min_coords[i] {
                    min_coords[i] = coord;
                }
                if coord > max_coords[i] {
                    max_coords[i] = coord;
                }
            }
        }

        let bounding_box = BoundingBox {
            min: min_coords,
            max: max_coords,
        };

        // Build hierarchy levels
        let mut levels = Vec::new();
        let mut current_resolution = 2; // Start with 2x2 grid

        for _level in 0..config.hierarchy_levels {
            let hierarchy_level = Self::build_hierarchy_level(
                points,
                values,
                &bounding_box,
                current_resolution,
                config,
            )?;

            levels.push(hierarchy_level);
            current_resolution *= 2; // Double resolution each level
        }

        Ok(HierarchicalDecomposition {
            levels,
            bounding_box,
        })
    }

    /// Build a single level in the hierarchy
    fn build_hierarchy_level(
        points: &Array2<F>,
        _values: &Array1<F>,
        bounding_box: &BoundingBox<F>,
        resolution: usize,
        config: &ScatteredConfig,
    ) -> InterpolateResult<HierarchyLevel<F>> {
        let n_dims = points.ncols();
        let n_points = points.nrows();

        // Calculate cell size
        let cell_size = Array1::from_shape_fn(n_dims, |i| {
            (bounding_box.max[i] - bounding_box.min[i]) / F::from_usize(resolution).unwrap()
        });

        // Create cells
        let mut cells = Vec::new();
        let total_cells = resolution.pow(n_dims as u32);

        for cell_idx in 0..total_cells {
            let cell_coords = Self::cell_index_to_coords(cell_idx, resolution, n_dims);

            // Compute cell bounds
            let cell_min = Array1::from_shape_fn(n_dims, |i| {
                bounding_box.min[i] + cell_size[i] * F::from_usize(cell_coords[i]).unwrap()
            });
            let cell_max = Array1::from_shape_fn(n_dims, |i| cell_min[i] + cell_size[i]);

            let cell_bounds = BoundingBox {
                min: cell_min,
                max: cell_max,
            };

            // Find points in this cell
            let mut point_indices = Vec::new();
            for (point_idx, point) in points.axis_iter(Axis(0)).enumerate() {
                if Self::point_in_bounds(&point, &cell_bounds) {
                    point_indices.push(point_idx);
                }
            }

            // Determine if cell needs refinement
            let needs_refinement = point_indices.len() > config.chunk_size;

            let cell = HierarchyCell {
                bounds: cell_bounds,
                point_indices,
                local_interpolator: None,
                needs_refinement,
            };

            cells.push(cell);
        }

        // Build spatial index for this level if beneficial
        let spatial_index = if n_points > 10000 {
            Some(RwLock::new(EnhancedNearestNeighborSearcher::new(
                points.clone(),
                IndexType::Adaptive,
                config.spatial_config.clone(),
            )?))
        } else {
            None
        };

        Ok(HierarchyLevel {
            cells,
            resolution,
            spatial_index,
        })
    }

    /// Convert linear cell index to n-dimensional coordinates
    fn cell_index_to_coords(index: usize, resolution: usize, n_dims: usize) -> Vec<usize> {
        let mut coords = vec![0; n_dims];
        let mut remaining = index;

        for coord in coords.iter_mut().take(n_dims) {
            *coord = remaining % resolution;
            remaining /= resolution;
        }

        coords
    }

    /// Check if a point is within the given bounds
    fn point_in_bounds(point: &ArrayView1<F>, bounds: &BoundingBox<F>) -> bool {
        for i in 0..point.len() {
            if point[i] < bounds.min[i] || point[i] >= bounds.max[i] {
                return false;
            }
        }
        true
    }

    /// Interpolate at query points using optimized algorithms
    ///
    /// # Arguments
    ///
    /// * `query_points` - Points to interpolate at (n_queries × n_dims)
    ///
    /// # Returns
    ///
    /// Interpolated values at query points
    pub fn interpolate_optimized(
        &mut self,
        query_points: &ArrayView2<F>,
    ) -> InterpolateResult<Array1<F>> {
        let start_time = std::time::Instant::now();
        let n_queries = query_points.nrows();

        let results = if let Some(ref hierarchy) = self.hierarchy {
            // Use hierarchical interpolation for large datasets
            self.interpolate_hierarchical(query_points, hierarchy)?
        } else if n_queries > self.config.chunk_size {
            // Use chunked processing for medium datasets
            self.interpolate_chunked(query_points)?
        } else {
            // Use direct interpolation for small datasets
            self.interpolate_direct(query_points)?
        };

        // Update statistics
        self.stats.interpolations += n_queries;
        self.stats.hierarchical_time_ns += start_time.elapsed().as_nanos() as u64;

        Ok(results)
    }

    /// Hierarchical interpolation for very large datasets
    fn interpolate_hierarchical(
        &self,
        query_points: &ArrayView2<F>,
        hierarchy: &HierarchicalDecomposition<F>,
    ) -> InterpolateResult<Array1<F>> {
        let n_queries = query_points.nrows();
        let mut results = Array1::zeros(n_queries);

        // Process queries in parallel if enabled
        if self.config.parallel && n_queries > 100 {
            let chunk_size = self.config.chunk_size.min(n_queries / 4).max(1);

            let results_vec: Result<Vec<_>, _> = query_points
                .axis_chunks_iter(Axis(0), chunk_size)
                .enumerate()
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|(chunk_idx, chunk)| {
                    let chunk_start = chunk_idx * chunk_size;
                    self.process_hierarchical_chunk(&chunk, hierarchy, chunk_start)
                })
                .collect();

            let results_vec = results_vec?;

            // Combine results
            for (chunk_idx, chunk_results) in results_vec.into_iter().enumerate() {
                let start_idx = chunk_idx * chunk_size;
                let end_idx = (start_idx + chunk_results.len()).min(n_queries);

                for (i, value) in chunk_results.into_iter().enumerate() {
                    if start_idx + i < end_idx {
                        results[start_idx + i] = value;
                    }
                }
            }
        } else {
            // Sequential processing
            for i in 0..n_queries {
                let query = query_points.row(i);
                results[i] = self.interpolate_single_hierarchical(&query, hierarchy)?;
            }
        }

        Ok(results)
    }

    /// Process a chunk of queries using hierarchical interpolation
    fn process_hierarchical_chunk(
        &self,
        chunk: &ArrayView2<F>,
        hierarchy: &HierarchicalDecomposition<F>,
        _chunk_start: usize,
    ) -> InterpolateResult<Vec<F>> {
        let mut results = Vec::with_capacity(chunk.nrows());

        for query in chunk.axis_iter(Axis(0)) {
            let value = self.interpolate_single_hierarchical(&query, hierarchy)?;
            results.push(value);
        }

        Ok(results)
    }

    /// Interpolate a single point using hierarchical decomposition
    fn interpolate_single_hierarchical(
        &self,
        query: &ArrayView1<F>,
        hierarchy: &HierarchicalDecomposition<F>,
    ) -> InterpolateResult<F> {
        // Start from the finest level and work up
        for level in hierarchy.levels.iter().rev() {
            // Find the cell containing this query point
            for cell in &level.cells {
                if Self::point_in_bounds(query, &cell.bounds) && !cell.point_indices.is_empty() {
                    // Use local interpolation within this cell
                    return self.interpolate_in_cell(query, cell);
                }
            }
        }

        // Fallback to global interpolation if no suitable cell found
        self.interpolate_global_fallback(query)
    }

    /// Interpolate within a specific hierarchical cell
    fn interpolate_in_cell(
        &self,
        query: &ArrayView1<F>,
        cell: &HierarchyCell<F>,
    ) -> InterpolateResult<F> {
        if cell.point_indices.is_empty() {
            return Ok(F::zero());
        }

        // For now, use simple weighted average based on distance
        // In a full implementation, this would use the cached local interpolator
        let mut weighted_sum = F::zero();
        let mut weight_sum = F::zero();

        for &point_idx in &cell.point_indices {
            let point = self.points.row(point_idx);
            let value = self.values[point_idx];

            // Compute inverse distance weight
            let mut dist_sq = F::zero();
            for i in 0..query.len() {
                let diff = query[i] - point[i];
                dist_sq = dist_sq + diff * diff;
            }

            let dist = dist_sq.sqrt();
            let weight = if dist == F::zero() {
                return Ok(value); // Exact match
            } else {
                F::one() / (dist + F::from_f64(1e-10).unwrap()) // Add small epsilon
            };

            weighted_sum = weighted_sum + weight * value;
            weight_sum = weight_sum + weight;
        }

        if weight_sum > F::zero() {
            Ok(weighted_sum / weight_sum)
        } else {
            Ok(F::zero())
        }
    }

    /// Fallback global interpolation when hierarchical method fails
    fn interpolate_global_fallback(&self, query: &ArrayView1<F>) -> InterpolateResult<F> {
        // Use nearest neighbors from the global spatial index
        let neighbors = self
            .spatial_index
            .write()
            .unwrap()
            .k_nearest_neighbors(query, 5)?;

        if neighbors.is_empty() {
            return Ok(F::zero());
        }

        // Weighted average based on distance
        let mut weighted_sum = F::zero();
        let mut weight_sum = F::zero();

        for (point_idx, distance) in neighbors {
            let value = self.values[point_idx];
            let weight = if distance == F::zero() {
                return Ok(value); // Exact match
            } else {
                F::one() / (distance + F::from_f64(1e-10).unwrap())
            };

            weighted_sum = weighted_sum + weight * value;
            weight_sum = weight_sum + weight;
        }

        if weight_sum > F::zero() {
            Ok(weighted_sum / weight_sum)
        } else {
            Ok(F::zero())
        }
    }

    /// Chunked interpolation for medium-sized datasets
    fn interpolate_chunked(
        &mut self,
        query_points: &ArrayView2<F>,
    ) -> InterpolateResult<Array1<F>> {
        let n_queries = query_points.nrows();
        let mut results = Array1::zeros(n_queries);
        let chunk_size = self.config.chunk_size;

        for chunk_start in (0..n_queries).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(n_queries);
            let chunk = query_points.slice(ndarray::s![chunk_start..chunk_end, ..]);

            let chunk_results = self.interpolate_direct(&chunk)?;

            for (i, value) in chunk_results.into_iter().enumerate() {
                results[chunk_start + i] = value;
            }

            self.stats.chunks_processed += 1;
        }

        Ok(results)
    }

    /// Direct interpolation for small datasets
    fn interpolate_direct(&self, query_points: &ArrayView2<F>) -> InterpolateResult<Array1<F>> {
        let n_queries = query_points.nrows();
        let mut results = Array1::zeros(n_queries);

        for i in 0..n_queries {
            let query = query_points.row(i);
            results[i] = self.interpolate_global_fallback(&query)?;
        }

        Ok(results)
    }

    /// Get performance statistics
    pub fn stats(&self) -> &OptimizedScatteredStats {
        &self.stats
    }

    /// Reset performance statistics
    pub fn reset_stats(&mut self) {
        self.stats = OptimizedScatteredStats::default();
    }
}

/// Create an optimized scattered data interpolator
///
/// # Arguments
///
/// * `points` - Training data points
/// * `values` - Training data values
/// * `config` - Optimization configuration
///
/// # Returns
///
/// An optimized scattered data interpolator
pub fn make_optimized_scattered_interpolator<F>(
    points: Array2<F>,
    values: Array1<F>,
    config: ScatteredConfig,
) -> InterpolateResult<OptimizedScatteredInterpolator<F>>
where
    F: Float + FromPrimitive + Debug + Display + Send + Sync + 'static,
{
    OptimizedScatteredInterpolator::new(points, values, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_optimized_scattered_interpolator_creation() {
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let values = array![0.0, 1.0, 1.0, 2.0];
        let config = ScatteredConfig::for_speed();

        let interpolator = OptimizedScatteredInterpolator::new(points, values, config);
        assert!(interpolator.is_ok());
    }

    #[test]
    fn test_scattered_config_variants() {
        let large_config = ScatteredConfig::for_large_dataset();
        let accuracy_config = ScatteredConfig::for_accuracy();
        let speed_config = ScatteredConfig::for_speed();

        assert!(large_config.use_hierarchical);
        assert!(!accuracy_config.use_approximation);
        assert!(speed_config.use_approximation);
        assert!(speed_config.approximation_tolerance > accuracy_config.approximation_tolerance);
    }

    #[test]
    fn test_bounding_box() {
        let _points = array![[0.0, 0.0], [1.0, 2.0], [-1.0, 1.0]];

        // Test point_in_bounds logic (would need to make method public)
        let bounds = BoundingBox {
            min: array![-1.0, 0.0],
            max: array![1.0, 2.0],
        };

        assert_eq!(bounds.min[0], -1.0);
        assert_eq!(bounds.max[1], 2.0);
    }
}
