//! Parallel implementation of Voronoi-based interpolation methods
//!
//! This module provides parallel versions of the natural neighbor interpolation methods,
//! which can significantly improve performance for large datasets or when interpolating
//! at many query points.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, FromPrimitive};
use rayon::prelude::*;
use std::fmt::Debug;
use std::sync::{Arc, Mutex};

use super::natural::{InterpolationMethod, NaturalNeighborInterpolator};
use super::voronoi_cell::VoronoiDiagram;
use crate::error::InterpolateResult;

/// Parallel implementation of Natural Neighbor interpolation
///
/// Uses Rayon's parallel iterators to parallelize interpolation of multiple query points.
#[derive(Debug, Clone)]
pub struct ParallelNaturalNeighborInterpolator<
    F: Float
        + FromPrimitive
        + Debug
        + Send
        + Sync
        + ndarray::ScalarOperand
        + 'static
        + std::cmp::PartialOrd,
> {
    /// The underlying sequential interpolator
    interpolator: NaturalNeighborInterpolator<F>,

    /// Configuration for parallel execution
    parallel_config: ParallelConfig,
}

/// Configuration for parallel execution
#[derive(Debug, Clone, Copy)]
pub struct ParallelConfig {
    /// Minimum number of query points per thread
    pub min_points_per_thread: usize,

    /// Target number of threads to use
    pub target_threads: Option<usize>,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        ParallelConfig {
            min_points_per_thread: 10,
            target_threads: None,
        }
    }
}

impl<
        F: Float
            + FromPrimitive
            + Debug
            + Send
            + Sync
            + ndarray::ScalarOperand
            + 'static
            + std::cmp::PartialOrd,
    > ParallelNaturalNeighborInterpolator<F>
{
    /// Creates a new parallel Natural Neighbor interpolator
    ///
    /// # Arguments
    /// * `points` - The input data points (rows = points, columns = dimensions)
    /// * `values` - The values at the data points
    /// * `method` - The interpolation method to use
    /// * `config` - Configuration for parallel execution (optional)
    ///
    /// # Returns
    /// A new parallel Natural Neighbor interpolator
    pub fn new(
        points: Array2<F>,
        values: Array1<F>,
        method: InterpolationMethod,
        config: Option<ParallelConfig>,
    ) -> InterpolateResult<Self> {
        // Create the underlying sequential interpolator
        let interpolator = NaturalNeighborInterpolator::new(points, values, method)?;

        // Use the provided config or the default
        let parallel_config = config.unwrap_or_default();

        Ok(ParallelNaturalNeighborInterpolator {
            interpolator,
            parallel_config,
        })
    }

    /// Interpolates at a single query point
    ///
    /// This simply delegates to the underlying sequential interpolator,
    /// as parallelization for a single point doesn't make sense.
    ///
    /// # Arguments
    /// * `query` - The query point
    ///
    /// # Returns
    /// The interpolated value at the query point
    pub fn interpolate(&self, query: &ArrayView1<F>) -> InterpolateResult<F> {
        self.interpolator.interpolate(query)
    }

    /// Interpolates at multiple query points in parallel
    ///
    /// # Arguments
    /// * `queries` - The query points (rows = points, columns = dimensions)
    ///
    /// # Returns
    /// An array of interpolated values
    pub fn interpolate_multi(&self, queries: &ArrayView2<F>) -> InterpolateResult<Array1<F>> {
        let n_queries = queries.nrows();
        let _dim = queries.ncols();

        // For very small numbers of queries, just use the sequential version
        if n_queries <= self.parallel_config.min_points_per_thread {
            return self.interpolator.interpolate_multi(queries);
        }

        // Determine the number of threads to use
        let num_threads = match self.parallel_config.target_threads {
            Some(n) => n,
            None => rayon::current_num_threads()
                .min(n_queries / self.parallel_config.min_points_per_thread),
        }
        .max(1);

        let chunk_size = n_queries.div_ceil(num_threads);

        // Create a container for the results
        let results = Arc::new(Mutex::new(Array1::zeros(n_queries)));

        // Process the queries in parallel
        (0..n_queries)
            .into_par_iter()
            .chunks(chunk_size)
            .try_for_each(|chunk| -> InterpolateResult<()> {
                for i in chunk {
                    let query = queries.row(i);

                    match self.interpolator.interpolate(&query) {
                        Ok(value) => {
                            let mut results = results.lock().unwrap();
                            results[i] = value;
                        }
                        Err(err) => return Err(err),
                    }
                }

                Ok(())
            })?;

        // Return the results
        Ok(Arc::try_unwrap(results).unwrap().into_inner().unwrap())
    }

    /// Returns the underlying interpolation method
    pub fn method(&self) -> InterpolationMethod {
        self.interpolator.method()
    }

    /// Sets the interpolation method
    pub fn set_method(&mut self, method: InterpolationMethod) {
        self.interpolator.set_method(method);
    }

    /// Updates the parallel configuration
    pub fn set_parallel_config(&mut self, config: ParallelConfig) {
        self.parallel_config = config;
    }

    /// Returns the underlying Voronoi diagram
    pub fn voronoi_diagram(&self) -> &VoronoiDiagram<F> {
        self.interpolator.voronoi_diagram()
    }
}

/// Creates a new parallel Natural Neighbor interpolator
///
/// # Arguments
/// * `points` - The input data points (rows = points, columns = dimensions)
/// * `values` - The values at the data points
/// * `method` - The interpolation method to use
/// * `config` - Configuration for parallel execution (optional)
///
/// # Returns
/// A new parallel Natural Neighbor interpolator
pub fn make_parallel_natural_neighbor_interpolator<
    F: Float + FromPrimitive + Debug + Send + Sync + ndarray::ScalarOperand + 'static + std::cmp::Ord,
>(
    points: Array2<F>,
    values: Array1<F>,
    method: InterpolationMethod,
    config: Option<ParallelConfig>,
) -> InterpolateResult<ParallelNaturalNeighborInterpolator<F>> {
    ParallelNaturalNeighborInterpolator::new(points, values, method, config)
}

/// Creates a new parallel Sibson interpolator
///
/// # Arguments
/// * `points` - The input data points (rows = points, columns = dimensions)
/// * `values` - The values at the data points
/// * `config` - Configuration for parallel execution (optional)
///
/// # Returns
/// A new parallel Natural Neighbor interpolator using Sibson's method
pub fn make_parallel_sibson_interpolator<
    F: Float
        + FromPrimitive
        + Debug
        + Send
        + Sync
        + ndarray::ScalarOperand
        + 'static
        + std::cmp::PartialOrd,
>(
    points: Array2<F>,
    values: Array1<F>,
    config: Option<ParallelConfig>,
) -> InterpolateResult<ParallelNaturalNeighborInterpolator<F>> {
    ParallelNaturalNeighborInterpolator::new(points, values, InterpolationMethod::Sibson, config)
}

/// Creates a new parallel Laplace interpolator
///
/// # Arguments
/// * `points` - The input data points (rows = points, columns = dimensions)
/// * `values` - The values at the data points
/// * `config` - Configuration for parallel execution (optional)
///
/// # Returns
/// A new parallel Natural Neighbor interpolator using the non-Sibsonian (Laplace) method
pub fn make_parallel_laplace_interpolator<
    F: Float
        + FromPrimitive
        + Debug
        + Send
        + Sync
        + ndarray::ScalarOperand
        + 'static
        + std::cmp::PartialOrd,
>(
    points: Array2<F>,
    values: Array1<F>,
    config: Option<ParallelConfig>,
) -> InterpolateResult<ParallelNaturalNeighborInterpolator<F>> {
    ParallelNaturalNeighborInterpolator::new(points, values, InterpolationMethod::Laplace, config)
}
