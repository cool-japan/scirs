//! Fast Approximate Kriging for Large Datasets
//!
//! This module provides computationally efficient kriging algorithms for large spatial datasets.
//! Standard kriging requires O(n³) operations for fitting and O(n²) for prediction,
//! which becomes prohibitively expensive for large datasets. This module implements:
//!
//! 1. **Local Kriging**: Uses only nearby points for each prediction
//! 2. **Fixed Rank Kriging**: Low-rank approximation of the covariance matrix
//! 3. **Sparse Cholesky**: Efficient factorization for large sparse matrices
//! 4. **Tapering**: Covariance tapering for sparse approximations
//!
//! These methods trade some accuracy for substantial performance improvements,
//! making kriging feasible for datasets with thousands to millions of points.

use crate::advanced::enhanced_kriging::{AnisotropicCovariance, TrendFunction};
use crate::advanced::kriging::CovarianceFunction;
use crate::error::{InterpolateError, InterpolateResult};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, FromPrimitive};
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Sub};

/// Type alias for sparse matrix representation
type SparseComponents<F> = (Vec<(usize, usize)>, Vec<F>);

/// Maximum number of neighbors to consider in local kriging
const DEFAULT_MAX_NEIGHBORS: usize = 50;

/// Default radius multiplier for local neighborhood search
const DEFAULT_RADIUS_MULTIPLIER: f64 = 3.0;

// Import submodules
pub mod ordinary;
pub mod universal;
pub mod variogram;
pub mod covariance;
pub mod acceleration;

// Re-export public items from submodules
pub use acceleration::*;
pub use covariance::*; 
pub use ordinary::*;
pub use universal::*;
pub use variogram::*;

/// Fast kriging approximation methods for large datasets
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FastKrigingMethod {
    /// Local kriging using only nearby points for prediction
    /// Provides O(k³) complexity per prediction where k is the neighborhood size
    Local,

    /// Fixed Rank Kriging with low-rank approximation
    /// Provides O(nr²) fitting and O(r²) prediction where r is the rank
    FixedRank(usize), // Rank parameter

    /// Tapering approach that zeros out small covariance values
    /// Creates sparse matrices for efficient computation
    Tapering(f64), // Taper range parameter

    /// Hierarchical off-diagonal low-rank approximation
    /// Balances accuracy and computational efficiency
    HODLR(usize), // Maximum leaf size
}

/// Fast approximate kriging interpolator for large datasets
///
/// This struct provides methods for efficient kriging interpolation
/// when dealing with large spatial datasets (thousands to millions of points).
/// It trades some accuracy for significant computational savings.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "linalg")]
/// # {
/// use ndarray::{Array1, Array2};
/// use scirs2_interpolate::advanced::fast_kriging::{
///     FastKriging, FastKrigingMethod, FastKrigingBuilder
/// };
/// use scirs2_interpolate::advanced::kriging::CovarianceFunction;
///
/// // Create sample data
/// let n_points = 100; // Reduced for testing
/// let points = Array2::<f64>::zeros((n_points, 2));
/// let values = Array1::<f64>::zeros(n_points);
///
/// // Create a fast kriging model using local approximation
/// let local_kriging = FastKrigingBuilder::<f64>::new()
///     .points(points.clone())
///     .values(values.clone())
///     .covariance_function(CovarianceFunction::Matern52)
///     .approximation_method(FastKrigingMethod::Local)
///     .max_neighbors(50)
///     .build()
///     .unwrap();
///
/// // Predict at new points
/// let query_points = Array2::<f64>::zeros((10, 2));
/// let predictions = local_kriging.predict(&query_points.view()).unwrap();
///
/// // Create a model using fixed rank approximation
/// let low_rank_kriging = FastKrigingBuilder::<f64>::new()
///     .points(points.clone())
///     .values(values.clone())
///     .covariance_function(CovarianceFunction::Exponential)
///     .approximation_method(FastKrigingMethod::FixedRank(10))
///     .build()
///     .unwrap();
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct FastKriging<F>
where
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + Div<Output = F>
        + Mul<Output = F>
        + Sub<Output = F>
        + Add<Output = F>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign,
{
    /// Points coordinates (input locations)
    points: Array2<F>,

    /// Values at points (observations)
    values: Array1<F>,

    /// Anisotropic covariance specification
    anisotropic_cov: AnisotropicCovariance<F>,

    /// Trend function type for Universal Kriging
    trend_fn: TrendFunction,

    /// Approximation method to use for fast computation
    approx_method: FastKrigingMethod,

    /// Maximum number of neighbors for local kriging
    max_neighbors: usize,

    /// Search radius multiplier for local kriging
    radius_multiplier: F,

    /// Pre-computed low-rank approximation components
    /// For FixedRank: [U, S, V] where K ≈ U * S * V^T
    low_rank_components: Option<(Array2<F>, Array1<F>, Array2<F>)>,

    /// Sparse representation for Tapering method
    /// Indices and values for sparse covariance matrix
    sparse_components: Option<SparseComponents<F>>,

    /// Weights for kriging predictions
    /// Different format based on approximation method
    weights: Array1<F>,

    /// Pre-computed basis functions for trend model
    basis_functions: Option<Array2<F>>,

    /// Trend coefficients for Universal Kriging
    trend_coeffs: Option<Array1<F>>,

    /// Flag indicating whether to optimize parameters
    optimize_parameters: bool,

    /// Flag indicating whether exact variance computation is needed
    compute_exact_variance: bool,

    /// Marker for generic type parameter
    _phantom: PhantomData<F>,
}

/// Builder for constructing FastKriging models
///
/// This builder provides a method-chaining interface for configuring and
/// creating FastKriging models for large datasets.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "linalg")]
/// # {
/// use ndarray::{Array1, Array2};
/// use scirs2_interpolate::advanced::fast_kriging::{
///     FastKrigingBuilder, FastKrigingMethod
/// };
/// use scirs2_interpolate::advanced::kriging::CovarianceFunction;
///
/// // Create sample data
/// let points = Array2::<f64>::zeros((100, 2));
/// let values = Array1::<f64>::zeros(100);
///
/// // Build a fast kriging model with local approximation
/// let kriging = FastKrigingBuilder::<f64>::new()
///     .points(points.clone())
///     .values(values.clone())
///     .covariance_function(CovarianceFunction::Matern52)
///     .approximation_method(FastKrigingMethod::Local)
///     .max_neighbors(30)
///     .radius_multiplier(2.5)
///     .build()
///     .unwrap();
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct FastKrigingBuilder<F>
where
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + Div<Output = F>
        + Mul<Output = F>
        + Sub<Output = F>
        + Add<Output = F>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign,
{
    /// Points coordinates
    points: Option<Array2<F>>,

    /// Values at points
    values: Option<Array1<F>>,

    /// Covariance function
    cov_fn: CovarianceFunction,

    /// Directional length scales
    length_scales: Option<Array1<F>>,

    /// Signal variance parameter
    sigma_sq: F,

    /// Nugget parameter
    nugget: F,

    /// Trend function type
    trend_fn: TrendFunction,

    /// Approximation method
    approx_method: FastKrigingMethod,

    /// Maximum number of neighbors
    max_neighbors: usize,

    /// Search radius multiplier
    radius_multiplier: F,

    /// Marker for generic type parameter
    _phantom: PhantomData<F>,
}

// Base implementation
impl<F> FastKriging<F>
where
    F: Float + FromPrimitive + Debug + Display + Add<Output = F> + Sub<Output = F> + Mul<Output = F> + Div<Output = F>
        + std::ops::AddAssign + std::ops::SubAssign + std::ops::MulAssign + std::ops::DivAssign + std::ops::RemAssign
        + 'static,
{
    /// Create a new FastKriging instance from a builder
    ///
    /// This is used internally by the builder and shouldn't be called directly.
    /// Use `FastKrigingBuilder::build()` instead.
    pub(crate) fn from_builder(builder: FastKrigingBuilder<F>) -> InterpolateResult<Self> {
        // Basic validation will be implemented in the ordinary and universal modules
        Ok(Self {
            points: builder.points.ok_or(InterpolateError::MissingPoints)?,
            values: builder.values.ok_or(InterpolateError::MissingValues)?,
            anisotropic_cov: AnisotropicCovariance::new(
                builder.cov_fn,
                builder.length_scales.unwrap_or_else(|| Array1::from_elem(1, F::one())),
                builder.sigma_sq,
                builder.nugget,
            ),
            trend_fn: builder.trend_fn,
            approx_method: builder.approx_method,
            max_neighbors: builder.max_neighbors,
            radius_multiplier: builder.radius_multiplier,
            low_rank_components: None,
            sparse_components: None,
            weights: Array1::zeros(0),
            basis_functions: None,
            trend_coeffs: None,
            optimize_parameters: false,
            compute_exact_variance: false,
            _phantom: PhantomData,
        })
    }
}

// Builder implementation
impl<F> Default for FastKrigingBuilder<F>
where
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
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F> FastKrigingBuilder<F>
where
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
{
    /// Create a new builder for FastKriging
    pub fn new() -> Self {
        Self {
            points: None,
            values: None,
            cov_fn: CovarianceFunction::Matern52,
            length_scales: None,
            sigma_sq: F::from_f64(1.0).unwrap(),
            nugget: F::from_f64(1e-6).unwrap(),
            trend_fn: TrendFunction::Constant,
            approx_method: FastKrigingMethod::Local,
            max_neighbors: DEFAULT_MAX_NEIGHBORS,
            radius_multiplier: F::from_f64(DEFAULT_RADIUS_MULTIPLIER).unwrap(),
            _phantom: PhantomData,
        }
    }
}