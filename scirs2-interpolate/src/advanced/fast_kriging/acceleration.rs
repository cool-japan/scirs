//! Acceleration Techniques for Fast Kriging
//!
//! This module provides computational acceleration methods for kriging
//! with large datasets, including spatial indexing, approximation techniques,
//! and parallel processing.

use crate::advanced::enhanced_kriging::AnisotropicCovariance;
use crate::advanced::fast_kriging::{FastKriging, FastKrigingBuilder, FastKrigingMethod};
use crate::advanced::kriging::CovarianceFunction;
use crate::error::InterpolateResult;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, FromPrimitive};
use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Mul, Sub};

/// Creates a new FastKriging model with local approximation
///
/// This convenience function provides a simpler interface for creating
/// a FastKriging model using the local approximation method, which is
/// suitable for most large datasets.
///
/// # Arguments
///
/// * `points` - Coordinates of sample points (n_points × n_dimensions)
/// * `values` - Values at sample points (n_points)
/// * `cov_fn` - Covariance function to use
/// * `length_scale` - Isotropic length scale parameter
/// * `max_neighbors` - Maximum number of neighbors to use
///
/// # Returns
///
/// A FastKriging interpolator with local approximation method
///
/// # Example
///
/// ```
/// # #[cfg(feature = "linalg")]
/// # {
/// use ndarray::{Array1, Array2};
/// use scirs2_interpolate::advanced::fast_kriging::make_local_kriging;
/// use scirs2_interpolate::advanced::kriging::CovarianceFunction;
///
/// // Create sample data
/// let points = Array2::<f64>::zeros((100, 2));
/// let values = Array1::<f64>::zeros(100);
///
/// // Create a local kriging model
/// let kriging = make_local_kriging(
///     &points.view(),
///     &values.view(),
///     CovarianceFunction::Matern52,
///     1.0,  // length_scale
///     50    // max_neighbors
/// ).unwrap();
///
/// // Make a prediction
/// let query_point = Array2::<f64>::zeros((1, 2));
/// let pred = kriging.predict(&query_point.view()).unwrap();
/// # }
/// ```
pub fn make_local_kriging<
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + Add<Output = F>
        + Sub<Output = F>
        + Mul<Output = F>
        + Div<Output = F>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign
        + 'static,
>(
    points: &ArrayView2<F>,
    values: &ArrayView1<F>,
    cov_fn: CovarianceFunction,
    length_scale: F,
    max_neighbors: usize,
) -> InterpolateResult<FastKriging<F>> {
    // Create length scales array with isotropic value
    let n_dims = points.shape()[1];
    let length_scales = Array1::from_elem(n_dims, length_scale);

    // Use builder to create the model
    FastKrigingBuilder::<F>::new()
        .points(points.to_owned())
        .values(values.to_owned())
        .covariance_function(cov_fn)
        .length_scales(length_scales)
        .approximation_method(FastKrigingMethod::Local)
        .max_neighbors(max_neighbors)
        .build()
}

/// Creates a new FastKriging model with fixed rank approximation
///
/// This function creates a kriging model that uses a low-rank approximation
/// of the covariance matrix to enable fast computation with large datasets.
///
/// # Arguments
///
/// * `points` - Coordinates of sample points (n_points × n_dimensions)
/// * `values` - Values at sample points (n_points)
/// * `cov_fn` - Covariance function to use
/// * `length_scale` - Isotropic length scale parameter
/// * `rank` - Rank of the approximation (smaller rank = faster but less accurate)
///
/// # Returns
///
/// A FastKriging interpolator with fixed rank approximation method
///
/// # Example
///
/// ```
/// # #[cfg(feature = "linalg")]
/// # {
/// use ndarray::{Array1, Array2};
/// use scirs2_interpolate::advanced::fast_kriging::make_fixed_rank_kriging;
/// use scirs2_interpolate::advanced::kriging::CovarianceFunction;
///
/// // Create sample data
/// let points = Array2::<f64>::zeros((100, 2));
/// let values = Array1::<f64>::zeros(100);
///
/// // Create a fixed rank kriging model
/// let kriging = make_fixed_rank_kriging(
///     &points.view(),
///     &values.view(),
///     CovarianceFunction::Matern52,
///     1.0,  // length_scale
///     10    // rank
/// ).unwrap();
///
/// // Make a prediction
/// let query_point = Array2::<f64>::zeros((1, 2));
/// let pred = kriging.predict(&query_point.view()).unwrap();
/// # }
/// ```
pub fn make_fixed_rank_kriging<
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + Add<Output = F>
        + Sub<Output = F>
        + Mul<Output = F>
        + Div<Output = F>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign
        + 'static,
>(
    points: &ArrayView2<F>,
    values: &ArrayView1<F>,
    cov_fn: CovarianceFunction,
    length_scale: F,
    rank: usize,
) -> InterpolateResult<FastKriging<F>> {
    // Create length scales array with isotropic value
    let n_dims = points.shape()[1];
    let length_scales = Array1::from_elem(n_dims, length_scale);

    // Use builder to create the model
    FastKrigingBuilder::<F>::new()
        .points(points.to_owned())
        .values(values.to_owned())
        .covariance_function(cov_fn)
        .length_scales(length_scales)
        .approximation_method(FastKrigingMethod::FixedRank(rank))
        .build()
}

/// Creates a new FastKriging model with tapering approximation
///
/// This function creates a kriging model that uses covariance tapering
/// to create a sparse representation, enabling fast computation with large datasets.
///
/// # Arguments
///
/// * `points` - Coordinates of sample points (n_points × n_dimensions)
/// * `values` - Values at sample points (n_points)
/// * `cov_fn` - Covariance function to use
/// * `length_scale` - Isotropic length scale parameter
/// * `taper_range` - Range beyond which covariance is set to zero
///
/// # Returns
///
/// A FastKriging interpolator with tapering approximation method
///
/// # Example
///
/// ```
/// # #[cfg(feature = "linalg")]
/// # {
/// use ndarray::{Array1, Array2};
/// use scirs2_interpolate::advanced::fast_kriging::make_tapered_kriging;
/// use scirs2_interpolate::advanced::kriging::CovarianceFunction;
///
/// // Create sample data
/// let points = Array2::<f64>::zeros((100, 2));
/// let values = Array1::<f64>::zeros(100);
///
/// // Create a tapered kriging model
/// let kriging = make_tapered_kriging(
///     &points.view(),
///     &values.view(),
///     CovarianceFunction::Matern52,
///     1.0,  // length_scale
///     3.0   // taper_range
/// ).unwrap();
///
/// // Make a prediction
/// let query_point = Array2::<f64>::zeros((1, 2));
/// let pred = kriging.predict(&query_point.view()).unwrap();
/// # }
/// ```
pub fn make_tapered_kriging<
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + Add<Output = F>
        + Sub<Output = F>
        + Mul<Output = F>
        + Div<Output = F>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign
        + 'static,
>(
    points: &ArrayView2<F>,
    values: &ArrayView1<F>,
    cov_fn: CovarianceFunction,
    length_scale: F,
    taper_range: F,
) -> InterpolateResult<FastKriging<F>> {
    // Create length scales array with isotropic value
    let n_dims = points.shape()[1];
    let length_scales = Array1::from_elem(n_dims, length_scale);

    // Use builder to create the model
    FastKrigingBuilder::<F>::new()
        .points(points.to_owned())
        .values(values.to_owned())
        .covariance_function(cov_fn)
        .length_scales(length_scales)
        .approximation_method(FastKrigingMethod::Tapering(taper_range.to_f64().unwrap()))
        .build()
}

/// Creates a new FastKriging model with HODLR approximation
///
/// This function creates a kriging model that uses a hierarchical off-diagonal
/// low-rank approximation for fast computation with large datasets.
///
/// # Arguments
///
/// * `points` - Coordinates of sample points (n_points × n_dimensions)
/// * `values` - Values at sample points (n_points)
/// * `cov_fn` - Covariance function to use
/// * `length_scale` - Isotropic length scale parameter
/// * `leaf_size` - Size of leaf nodes in the hierarchical decomposition
///
/// # Returns
///
/// A FastKriging interpolator with HODLR approximation method
///
/// # Example
///
/// ```
/// # #[cfg(feature = "linalg")]
/// # {
/// use ndarray::{Array1, Array2};
/// use scirs2_interpolate::advanced::fast_kriging::make_hodlr_kriging;
/// use scirs2_interpolate::advanced::kriging::CovarianceFunction;
///
/// // Create sample data
/// let points = Array2::<f64>::zeros((100, 2));
/// let values = Array1::<f64>::zeros(100);
///
/// // Create a HODLR kriging model
/// let kriging = make_hodlr_kriging(
///     &points.view(),
///     &values.view(),
///     CovarianceFunction::Matern52,
///     1.0,  // length_scale
///     32    // leaf_size
/// ).unwrap();
///
/// // Make a prediction
/// let query_point = Array2::<f64>::zeros((1, 2));
/// let pred = kriging.predict(&query_point.view()).unwrap();
/// # }
/// ```
pub fn make_hodlr_kriging<
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + Add<Output = F>
        + Sub<Output = F>
        + Mul<Output = F>
        + Div<Output = F>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign
        + 'static,
>(
    points: &ArrayView2<F>,
    values: &ArrayView1<F>,
    cov_fn: CovarianceFunction,
    length_scale: F,
    leaf_size: usize,
) -> InterpolateResult<FastKriging<F>> {
    // Create length scales array with isotropic value
    let n_dims = points.shape()[1];
    let length_scales = Array1::from_elem(n_dims, length_scale);

    // Use builder to create the model
    FastKrigingBuilder::<F>::new()
        .points(points.to_owned())
        .values(values.to_owned())
        .covariance_function(cov_fn)
        .length_scales(length_scales)
        .approximation_method(FastKrigingMethod::HODLR(leaf_size))
        .build()
}

/// Automatically choose the best approximation method based on dataset size
///
/// This function selects an appropriate approximation method based on
/// the size of the dataset, balancing accuracy and computational efficiency.
///
/// # Returns
///
/// A FastKrigingMethod appropriate for the given dataset size
///
/// # Example
///
/// ```
/// use scirs2_interpolate::advanced::fast_kriging::select_approximation_method;
///
/// // Get recommended method for a dataset with 10,000 points
/// let method = select_approximation_method(10_000);
/// ```
pub fn select_approximation_method(n_points: usize) -> FastKrigingMethod {
    if n_points < 500 {
        // For small datasets, local kriging is accurate and fast enough
        FastKrigingMethod::Local
    } else if n_points < 5_000 {
        // For medium datasets, use fixed rank with moderate rank
        FastKrigingMethod::FixedRank(50)
    } else if n_points < 50_000 {
        // For large datasets, use tapering with moderate range
        FastKrigingMethod::Tapering(3.0)
    } else {
        // For very large datasets, use HODLR with appropriate leaf size
        FastKrigingMethod::HODLR(64)
    }
}

/// Structure to track computational performance
#[derive(Debug, Clone)]
pub struct KrigingPerformanceStats {
    /// Total time taken for fitting (milliseconds)
    pub fit_time_ms: f64,
    
    /// Average time per prediction (milliseconds)
    pub predict_time_ms: f64,
    
    /// Number of data points
    pub n_points: usize,
    
    /// Number of dimensions
    pub n_dims: usize,
    
    /// Approximation method used
    pub method: FastKrigingMethod,
}

/// Benchmark different approximation methods on a dataset
///
/// This function evaluates the performance of different kriging approximation
/// methods on a given dataset, helping users select the most appropriate method.
///
/// # Arguments
///
/// * `points` - Coordinates of sample points (n_points × n_dimensions)
/// * `values` - Values at sample points (n_points)
/// * `cov_fn` - Covariance function to use
/// * `length_scale` - Isotropic length scale parameter
///
/// # Returns
///
/// A vector of performance statistics for different methods
///
/// # Example
///
/// ```
/// # #[cfg(feature = "linalg")]
/// # {
/// use ndarray::{Array1, Array2};
/// use scirs2_interpolate::advanced::fast_kriging::benchmark_methods;
/// use scirs2_interpolate::advanced::kriging::CovarianceFunction;
///
/// // Create sample data
/// let points = Array2::<f64>::zeros((100, 2));
/// let values = Array1::<f64>::zeros(100);
///
/// // Benchmark performance
/// let performance = benchmark_methods(
///     &points.view(),
///     &values.view(),
///     CovarianceFunction::Matern52,
///     1.0  // length_scale
/// );
///
/// for stat in performance {
///     println!("Method: {:?}, Fit time: {:.2}ms, Predict time: {:.2}ms",
///              stat.method, stat.fit_time_ms, stat.predict_time_ms);
/// }
/// # }
/// ```
#[cfg(feature = "std")]
pub fn benchmark_methods<
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + Add<Output = F>
        + Sub<Output = F>
        + Mul<Output = F>
        + Div<Output = F>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign
        + 'static,
>(
    points: &ArrayView2<F>,
    values: &ArrayView1<F>,
    cov_fn: CovarianceFunction,
    length_scale: F,
) -> Vec<KrigingPerformanceStats> {
    use std::time::Instant;
    
    let n_points = points.shape()[0];
    let n_dims = points.shape()[1];
    
    // Methods to benchmark
    let methods = vec![
        FastKrigingMethod::Local,
        FastKrigingMethod::FixedRank(std::cmp::min(50, n_points / 10)),
        FastKrigingMethod::Tapering(3.0),
        FastKrigingMethod::HODLR(32),
    ];
    
    // Create length scales array with isotropic value
    let length_scales = Array1::from_elem(n_dims, length_scale);
    
    // Create a sample of query points for prediction
    let n_query = std::cmp::min(100, n_points / 10);
    let mut query_points = Array2::zeros((n_query, n_dims));
    
    // Use a simple strategy to sample query points from the original dataset
    let stride = n_points / n_query;
    for i in 0..n_query {
        let idx = i * stride;
        query_points.slice_mut(ndarray::s![i, ..]).assign(&points.slice(ndarray::s![idx, ..]));
    }
    
    // Track performance
    let mut stats = Vec::new();
    
    for method in methods {
        // Build model and track time
        let start = Instant::now();
        
        let model_result = FastKrigingBuilder::<F>::new()
            .points(points.to_owned())
            .values(values.to_owned())
            .covariance_function(cov_fn)
            .length_scales(length_scales.clone())
            .approximation_method(method)
            .build();
            
        let fit_time = start.elapsed().as_secs_f64() * 1000.0;
        
        if let Ok(model) = model_result {
            // Predict and track time
            let start = Instant::now();
            
            let _ = model.predict(&query_points.view());
            
            let predict_time = start.elapsed().as_secs_f64() * 1000.0 / n_query as f64;
            
            stats.push(KrigingPerformanceStats {
                fit_time_ms: fit_time,
                predict_time_ms: predict_time,
                n_points,
                n_dims,
                method,
            });
        }
    }
    
    stats
}