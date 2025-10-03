//! Fast Approximate Kriging for Large Datasets
//!
//! This module provides computationally efficient kriging algorithms for large spatial datasets.

use crate::advanced::enhanced_kriging::{AnisotropicCovariance, TrendFunction};
use crate::advanced::kriging::CovarianceFunction;
use crate::error::{InterpolateError, InterpolateResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Sub};

/// Type alias for sparse matrix representation
type SparseComponents<F> = (Vec<(usize, usize)>, Vec<F>);

/// Maximum number of neighbors to consider in local kriging
const DEFAULT_MAX_NEIGHBORS: usize = 50;

/// Default radius multiplier for local neighborhood search
const DEFAULT_RADIUS_MULTIPLIER: f64 = 3.0;

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

/// Result type for FastKriging predictions
#[derive(Debug, Clone)]
pub struct FastPredictionResult<F: Float> {
    /// Predicted values
    pub value: Array1<F>,

    /// Approximate prediction variances
    pub variance: Array1<F>,

    /// Method used for computation
    pub method: FastKrigingMethod,

    /// Computation time in milliseconds (if available)
    pub computation_time_ms: Option<f64>,
}

/// Fast approximate kriging interpolator for large datasets
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

// Re-export everything from the original module
// through the fast_kriging_reexports module
pub use super::fast_kriging_reexports::*;
