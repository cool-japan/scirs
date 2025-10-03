//! Parallel processing utilities for interpolation
//!
//! This module provides parallel implementations of computationally intensive
//! interpolation operations. It enables efficient utilization of multi-core
//! processors to accelerate interpolation tasks.
//!
//! The module includes:
//!
//! - Parallel versions of local interpolation methods
//!   - `ParallelMovingLeastSquares`: Parallel implementation of MLS interpolation
//!   - `ParallelLocalPolynomialRegression`: Parallel implementation of LOESS
//! - Thread pool management for parallel operations
//! - Utility functions for work distribution and threading
//!
//! Most parallel implementations in this module use Rayon, which provides
//! work-stealing thread pools and parallel iterators.
//!
//! # Examples
//!
//! ```
//! use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
//! use scirs2_interpolate::parallel::{
//!     ParallelMovingLeastSquares, ParallelConfig,
//!     ParallelEvaluate
//! };
//! use scirs2_interpolate::local::mls::{WeightFunction, PolynomialBasis};
//!
//! // Create sample data
//! let points = Array2::from_shape_vec((5, 2), vec![
//!     0.0, 0.0,
//!     1.0, 0.0,
//!     0.0, 1.0,
//!     1.0, 1.0,
//!     0.5, 0.5,
//! ]).unwrap();
//! let values = Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0, 1.5]);
//!
//! // Create parallel MLS interpolator
//! let parallel_mls = ParallelMovingLeastSquares::new(
//!     points.clone(),
//!     values.clone(),
//!     WeightFunction::Gaussian,
//!     PolynomialBasis::Linear,
//!     0.5, // bandwidth
//! ).unwrap();
//!
//! // Evaluate at multiple points in parallel
//! let query_points = Array2::from_shape_vec((2, 2), vec![
//!     0.25, 0.25,
//!     0.75, 0.75,
//! ]).unwrap();
//!
//! let config = ParallelConfig::new();
//! let results: Array1<f64> = parallel_mls.evaluate_parallel(&query_points.view(), &config).unwrap();
//! ```

use scirs2_core::ndarray::{Array1, ArrayView2};
use scirs2_core::numeric::Float;
use scirs2_core::parallel_ops::*;
use std::fmt::Debug;

use crate::error::InterpolateResult;

/// Configuration for parallel execution
#[derive(Debug, Clone, Copy, Default)]
pub struct ParallelConfig {
    /// Number of worker threads to use
    /// If None, uses Rayon's default (usually the number of logical CPUs)
    pub n_workers: Option<usize>,

    /// Chunk size for parallel iterators
    /// If None, Rayon chooses automatically
    pub chunk_size: Option<usize>,
}

impl ParallelConfig {
    /// Create a new ParallelConfig with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of worker threads
    pub fn with_workers(mut self, nworkers: usize) -> Self {
        self.n_workers = Some(nworkers);
        self
    }

    /// Set the chunk size for parallel iterators
    pub fn with_chunk_size(mut self, chunksize: usize) -> Self {
        self.chunk_size = Some(chunksize);
        self
    }

    /// Thread pool initialization is now handled globally by scirs2-core
    /// This method is kept for compatibility but no longer creates a new pool
    pub fn init_thread_pool(&self) -> InterpolateResult<()> {
        // Thread pool configuration is now handled globally by scirs2-core
        // The n_workers parameter is preserved for future use but currently ignored
        Ok(())
    }

    /// Get the chunk size to use for a given total size
    pub fn get_chunk_size(&self, totalsize: usize) -> usize {
        match self.chunk_size {
            Some(_size) => _size,
            None => {
                // Choose a reasonable chunk _size based on total _size
                // This is a heuristic and might need tuning for different workloads
                let n_cpus = num_cpus::get();
                let min_chunks_per_cpu = 4; // Ensure at least 4 chunks per CPU for load balancing

                std::cmp::max(1, totalsize / (n_cpus * min_chunks_per_cpu))
            }
        }
    }
}

/// Trait for functions that can be parallelized
pub trait Parallelizable<T, R> {
    /// Execute the function in parallel on a batch of inputs
    fn execute_parallel(&self, inputs: &[T], config: &ParallelConfig) -> Vec<R>;
}

/// Implementation for any function that can be applied to a single input
impl<T, R, F> Parallelizable<T, R> for F
where
    F: Fn(&T) -> R + Sync,
    T: Sync,
    R: Send,
{
    fn execute_parallel(&self, inputs: &[T], config: &ParallelConfig) -> Vec<R> {
        let chunk_size = config.get_chunk_size(inputs.len());

        inputs
            .par_iter()
            .with_min_len(chunk_size)
            .map(self)
            .collect()
    }
}

/// Trait for types that support parallel evaluation
pub trait ParallelEvaluate<F: Float, O> {
    /// Evaluate at multiple points in parallel
    fn evaluate_parallel(
        &self,
        points: &ArrayView2<F>,
        config: &ParallelConfig,
    ) -> InterpolateResult<O>;
}

/// Trait for interpolators that can make batch predictions in parallel
pub trait ParallelPredict<F: Float> {
    /// Make predictions at multiple points in parallel
    fn predict_parallel(
        &self,
        points: &ArrayView2<F>,
        config: &ParallelConfig,
    ) -> InterpolateResult<Array1<F>>;
}

/// Helper function to estimate optimal chunk size
///
/// This function determines a reasonable chunk size for parallel processing
/// based on the total size and the computational cost of each item.
///
/// # Arguments
///
/// * `total_size` - Total number of items to process
/// * `costfactor` - Relative computational cost per item (higher = more expensive)
/// * `config` - Parallel configuration
///
/// # Returns
///
/// The recommended chunk size
#[allow(dead_code)]
pub fn estimate_chunk_size(total_size: usize, costfactor: f64, config: &ParallelConfig) -> usize {
    // If chunk _size is explicitly specified, use that
    if let Some(size) = config.chunk_size {
        return size;
    }

    // Otherwise, compute a reasonable chunk _size
    let n_cpus = match config.n_workers {
        Some(n) => n,
        None => num_cpus::get(),
    };

    // Base estimate on desired chunks per CPU
    let desired_chunks_per_cpu = if costfactor > 10.0 {
        // Expensive operations - fewer, larger chunks
        2
    } else if costfactor > 1.0 {
        // Moderately expensive - balanced
        4
    } else {
        // Cheap operations - more, smaller chunks for better load balancing
        8
    };

    let base_chunk_size = std::cmp::max(1, total_size / (n_cpus * desired_chunks_per_cpu));

    // Apply cost _factor adjustment
    let adjusted_size = (base_chunk_size as f64 * costfactor.sqrt()).ceil() as usize;

    // Ensure a reasonable bound
    std::cmp::min(total_size, std::cmp::max(1, adjusted_size))
}

/// Create evenly distributed indices for parallel tasks
///
/// This function partitions a range of indices into approximately
/// equal-sized chunks for distribution across worker threads.
///
/// # Arguments
///
/// * `total_size` - Total number of items
/// * `n_parts` - Number of partitions to create
///
/// # Returns
///
/// Vector of (start, end) index pairs for each partition
#[allow(dead_code)]
pub fn create_index_ranges(total_size: usize, nparts: usize) -> Vec<(usize, usize)> {
    if total_size == 0 || nparts == 0 {
        return Vec::new();
    }

    let n_parts = std::cmp::min(nparts, total_size);
    let mut ranges = Vec::with_capacity(n_parts);

    let chunk_size = total_size / n_parts;
    let remainder = total_size % n_parts;

    let mut start = 0;

    for i in 0..n_parts {
        let extra = if i < remainder { 1 } else { 0 };
        let end = start + chunk_size + extra;

        ranges.push((start, end));
        start = end;
    }

    ranges
}

pub mod loess;
pub mod mls;

pub use loess::{
    make_parallel_loess, make_parallel_robust_loess, ParallelLocalPolynomialRegression,
};
pub use mls::{make_parallel_mls, ParallelMovingLeastSquares};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_execution() {
        let numbers = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let config = ParallelConfig::new();

        // Define a function to square a number
        let square = |x: &i32| x * x;

        // Execute in parallel
        let result = square.execute_parallel(&numbers, &config);

        // Check results
        assert_eq!(result, vec![1, 4, 9, 16, 25, 36, 49, 64, 81, 100]);
    }

    #[test]
    fn test_index_ranges() {
        // Test with evenly divisible size
        let ranges = create_index_ranges(10, 5);
        assert_eq!(ranges, vec![(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]);

        // Test with remainder
        let ranges = create_index_ranges(11, 3);
        assert_eq!(ranges, vec![(0, 4), (4, 8), (8, 11)]);

        // Test with more parts than items
        let ranges = create_index_ranges(3, 5);
        assert_eq!(ranges, vec![(0, 1), (1, 2), (2, 3)]);
    }

    #[test]
    fn test_chunk_size_estimation() {
        let config = ParallelConfig::new();

        // Test with different cost factors
        let size_cheap = estimate_chunk_size(1000, 0.5, &config);
        let size_moderate = estimate_chunk_size(1000, 5.0, &config);
        let size_expensive = estimate_chunk_size(1000, 20.0, &config);

        // Expensive operations should have larger chunks
        assert!(size_expensive >= size_moderate);
        assert!(size_moderate >= size_cheap);

        // Test with explicit chunk size
        let config_explicit = ParallelConfig::new().with_chunk_size(42);
        let size = estimate_chunk_size(1000, 1.0, &config_explicit);
        assert_eq!(size, 42);
    }
}
