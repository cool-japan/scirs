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

/// Result of a fast kriging prediction
#[derive(Debug, Clone)]
pub struct FastPredictionResult<F>
where
    F: Float + FromPrimitive + Debug + Display,
{
    /// Predicted values at query points
    pub value: Array1<F>,
    
    /// Prediction variances at query points
    pub variance: Array1<F>,
    
    /// Approximation method used
    pub method: FastKrigingMethod,
    
    /// Computation time in milliseconds (if measured)
    pub computation_time_ms: Option<f64>,
}

impl<F> FastPredictionResult<F>
where
    F: Float + FromPrimitive + Debug + Display,
{
    /// Get the number of predictions
    pub fn len(&self) -> usize {
        self.value.len()
    }

    /// Check if the result is empty
    pub fn is_empty(&self) -> bool {
        self.value.is_empty()
    }

    /// Get the predicted values
    pub fn values(&self) -> &Array1<F> {
        &self.value
    }

    /// Get the prediction variances
    pub fn variances(&self) -> &Array1<F> {
        &self.variance
    }

    /// Get the standard deviations (sqrt of variances)
    pub fn standard_deviations(&self) -> Array1<F> {
        self.variance.mapv(|v| v.sqrt())
    }

    /// Get confidence intervals for the predictions
    pub fn confidence_intervals(&self, confidence_level: f64) -> InterpolateResult<Array2<F>> {
        if confidence_level <= 0.0 || confidence_level >= 1.0 {
            return Err(InterpolateError::InvalidValue(
                "Confidence level must be between 0 and 1".to_string(),
            ));
        }

        // Approximate z-score for normal distribution
        let z_score = F::from_f64(match confidence_level {
            level if level > 0.99 => 2.576,  // 99%
            level if level > 0.95 => 1.96,   // 95%
            level if level > 0.90 => 1.645,  // 90%
            level if level > 0.80 => 1.282,  // 80%
            _ => 1.96,  // Default to 95%
        }).unwrap();

        let std_devs = self.standard_deviations();
        let mut intervals = Array2::zeros((self.len(), 2));

        for i in 0..self.len() {
            let margin = z_score * std_devs[i];
            intervals[[i, 0]] = self.value[i] - margin; // Lower bound
            intervals[[i, 1]] = self.value[i] + margin; // Upper bound
        }

        Ok(intervals)
    }
}

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
        let points = builder.points.ok_or(InterpolateError::MissingPoints)?;
        let values = builder.values.ok_or(InterpolateError::MissingValues)?;
        
        if points.nrows() != values.len() {
            return Err(InterpolateError::DimensionMismatch(
                "Number of points must match number of values".to_string(),
            ));
        }

        // Create anisotropic covariance
        let anisotropic_cov = AnisotropicCovariance::new(
            builder.cov_fn,
            builder.length_scales.unwrap_or_else(|| Array1::from_elem(points.ncols(), F::one())),
            builder.sigma_sq,
            builder.nugget,
        );

        let mut kriging = Self {
            points,
            values,
            anisotropic_cov,
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
        };

        // Pre-compute components based on approximation method
        kriging.initialize_approximation()?;

        Ok(kriging)
    }

    /// Initialize approximation-specific components
    fn initialize_approximation(&mut self) -> InterpolateResult<()> {
        match self.approx_method {
            FastKrigingMethod::FixedRank(rank) => {
                let (u, s, v) = covariance::compute_low_rank_approximation(
                    &self.points,
                    &self.anisotropic_cov,
                    rank,
                )?;
                self.low_rank_components = Some((u, s, v));
            }
            FastKrigingMethod::Tapering(range) => {
                let sparse_components = covariance::compute_tapered_covariance(
                    &self.points,
                    &self.anisotropic_cov,
                    F::from_f64(range).unwrap(),
                )?;
                self.sparse_components = Some(sparse_components);
            }
            _ => {
                // No pre-computation needed for Local and HODLR
            }
        }

        // Compute basis functions and trend coefficients if needed
        if self.trend_fn != crate::advanced::enhanced_kriging::TrendFunction::Constant {
            let basis = universal::create_basis_functions(&self.points.view(), self.trend_fn)?;
            let coeffs = universal::compute_trend_coefficients(
                &self.points,
                &self.values,
                &basis,
                self.trend_fn,
            )?;
            self.basis_functions = Some(basis);
            self.trend_coeffs = Some(coeffs);
        }

        Ok(())
    }

    /// Predict values at query points
    pub fn predict(&self, query_points: &ArrayView2<F>) -> InterpolateResult<FastPredictionResult<F>> {
        if query_points.ncols() != self.points.ncols() {
            return Err(InterpolateError::DimensionMismatch(
                "Query points must have same dimensionality as training points".to_string(),
            ));
        }

        match self.approx_method {
            FastKrigingMethod::Local => self.predict_local(query_points),
            FastKrigingMethod::FixedRank(_) => self.predict_fixed_rank(query_points),
            FastKrigingMethod::Tapering(_) => self.predict_tapered(query_points),
            FastKrigingMethod::HODLR(_) => self.predict_hodlr(query_points),
        }
    }

    /// Get the number of training points
    pub fn n_points(&self) -> usize {
        self.points.nrows()
    }

    /// Get the dimensionality
    pub fn n_dims(&self) -> usize {
        self.points.ncols()
    }

    /// Get the approximation method
    pub fn approximation_method(&self) -> FastKrigingMethod {
        self.approx_method
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

    /// Set the training points
    pub fn points(mut self, points: Array2<F>) -> Self {
        self.points = Some(points);
        self
    }

    /// Set the training values
    pub fn values(mut self, values: Array1<F>) -> Self {
        self.values = Some(values);
        self
    }

    /// Set the covariance function
    pub fn covariance_function(mut self, cov_fn: CovarianceFunction) -> Self {
        self.cov_fn = cov_fn;
        self
    }

    /// Set the length scales
    pub fn length_scales(mut self, length_scales: Array1<F>) -> Self {
        self.length_scales = Some(length_scales);
        self
    }

    /// Set the signal variance parameter
    pub fn sigma_sq(mut self, sigma_sq: F) -> Self {
        self.sigma_sq = sigma_sq;
        self
    }

    /// Set the nugget parameter
    pub fn nugget(mut self, nugget: F) -> Self {
        self.nugget = nugget;
        self
    }

    /// Set the trend function
    pub fn trend_function(mut self, trend_fn: TrendFunction) -> Self {
        self.trend_fn = trend_fn;
        self
    }

    /// Set the approximation method
    pub fn approximation_method(mut self, method: FastKrigingMethod) -> Self {
        self.approx_method = method;
        self
    }

    /// Set the maximum number of neighbors for local kriging
    pub fn max_neighbors(mut self, max_neighbors: usize) -> Self {
        self.max_neighbors = max_neighbors;
        self
    }

    /// Set the radius multiplier for local kriging
    pub fn radius_multiplier(mut self, radius_multiplier: F) -> Self {
        self.radius_multiplier = radius_multiplier;
        self
    }

    /// Build the FastKriging model
    pub fn build(self) -> InterpolateResult<FastKriging<F>> {
        FastKriging::from_builder(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};
    use crate::advanced::kriging::CovarianceFunction;

    fn create_test_data(n_points: usize, n_dims: usize) -> (Array2<f64>, Array1<f64>) {
        let mut points = Array2::zeros((n_points, n_dims));
        let mut values = Array1::zeros(n_points);

        // Generate a simple test dataset with a known function
        for i in 0..n_points {
            for d in 0..n_dims {
                points[[i, d]] = (i as f64) / (n_points as f64) + (d as f64) * 0.1;
            }
            // Simple quadratic function
            let x = points[[i, 0]];
            let y = if n_dims > 1 { points[[i, 1]] } else { 0.0 };
            values[i] = x * x + y * y + 0.1 * x * y;
        }

        (points, values)
    }

    #[test]
    fn test_fast_prediction_result_methods() {
        let value = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let variance = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let result = FastPredictionResult {
            value,
            variance,
            method: FastKrigingMethod::Local,
            computation_time_ms: Some(100.0),
        };

        assert_eq!(result.len(), 3);
        assert!(!result.is_empty());
        assert_eq!(result.values().len(), 3);
        assert_eq!(result.variances().len(), 3);

        let std_devs = result.standard_deviations();
        assert!((std_devs[0] - 0.1_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_confidence_intervals() {
        let value = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let variance = Array1::from_vec(vec![0.01, 0.04, 0.09]);
        let result = FastPredictionResult {
            value,
            variance,
            method: FastKrigingMethod::Local,
            computation_time_ms: None,
        };

        let intervals = result.confidence_intervals(0.95).unwrap();
        assert_eq!(intervals.nrows(), 3);
        assert_eq!(intervals.ncols(), 2);

        // Check that intervals are valid (lower < upper)
        for i in 0..3 {
            assert!(intervals[[i, 0]] < intervals[[i, 1]]);
        }

        // Test invalid confidence level
        assert!(result.confidence_intervals(1.1).is_err());
        assert!(result.confidence_intervals(-0.1).is_err());
    }

    #[test]
    fn test_fast_kriging_method_variants() {
        assert_eq!(FastKrigingMethod::Local, FastKrigingMethod::Local);
        assert_ne!(FastKrigingMethod::Local, FastKrigingMethod::FixedRank(10));
        
        let method1 = FastKrigingMethod::FixedRank(10);
        let method2 = FastKrigingMethod::FixedRank(10);
        let method3 = FastKrigingMethod::FixedRank(20);
        
        assert_eq!(method1, method2);
        assert_ne!(method1, method3);
    }

    #[test]
    fn test_fast_kriging_builder_default() {
        let builder = FastKrigingBuilder::<f64>::new();
        
        // Test that defaults are reasonable
        assert!(builder.points.is_none());
        assert!(builder.values.is_none());
        assert_eq!(builder.cov_fn, CovarianceFunction::Matern52);
        assert_eq!(builder.approx_method, FastKrigingMethod::Local);
        assert_eq!(builder.max_neighbors, DEFAULT_MAX_NEIGHBORS);
    }

    #[test]
    fn test_fast_kriging_builder_methods() {
        let (points, values) = create_test_data(10, 2);
        let length_scales = Array1::from_vec(vec![1.0, 1.5]);

        let builder = FastKrigingBuilder::<f64>::new()
            .points(points.clone())
            .values(values.clone())
            .covariance_function(CovarianceFunction::Exponential)
            .length_scales(length_scales.clone())
            .sigma_sq(2.0)
            .nugget(0.01)
            .approximation_method(FastKrigingMethod::FixedRank(5))
            .max_neighbors(15)
            .radius_multiplier(2.0);

        // Verify the builder values
        assert!(builder.points.is_some());
        assert!(builder.values.is_some());
        assert_eq!(builder.cov_fn, CovarianceFunction::Exponential);
        assert_eq!(builder.sigma_sq, 2.0);
        assert_eq!(builder.nugget, 0.01);
        assert_eq!(builder.max_neighbors, 15);
        assert_eq!(builder.radius_multiplier, 2.0);
    }

    #[test] 
    fn test_fast_kriging_build_missing_data() {
        let builder = FastKrigingBuilder::<f64>::new();
        
        // Should fail without points
        let result = builder.build();
        assert!(result.is_err());
        
        let (points, _) = create_test_data(5, 2);
        let builder = FastKrigingBuilder::<f64>::new().points(points);
        
        // Should fail without values
        let result = builder.build();
        assert!(result.is_err());
    }

    #[test]
    fn test_fast_kriging_dimension_mismatch() {
        let points = Array2::zeros((5, 2));
        let values = Array1::zeros(3); // Wrong size
        
        let result = FastKrigingBuilder::<f64>::new()
            .points(points)
            .values(values)
            .build();
            
        assert!(result.is_err());
    }

    #[cfg(feature = "linalg")]
    #[test]
    fn test_local_kriging() {
        let (points, values) = create_test_data(20, 2);
        
        let kriging = FastKrigingBuilder::<f64>::new()
            .points(points.clone())
            .values(values.clone())
            .covariance_function(CovarianceFunction::Matern52)
            .approximation_method(FastKrigingMethod::Local)
            .max_neighbors(10)
            .build()
            .unwrap();

        // Test basic properties
        assert_eq!(kriging.n_points(), 20);
        assert_eq!(kriging.n_dims(), 2);
        assert_eq!(kriging.approximation_method(), FastKrigingMethod::Local);

        // Test prediction
        let query_points = Array2::from_shape_vec(
            (2, 2),
            vec![0.25, 0.25, 0.75, 0.75],
        ).unwrap();

        let result = kriging.predict(&query_points.view()).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result.method, FastKrigingMethod::Local);
        
        // Predictions should be reasonable
        for &val in result.values().iter() {
            assert!(val.is_finite());
        }
        
        // Variances should be non-negative
        for &var in result.variances().iter() {
            assert!(var >= 0.0);
        }
    }

    #[cfg(feature = "linalg")]
    #[test]
    fn test_fixed_rank_kriging() {
        let (points, values) = create_test_data(30, 2);
        
        let kriging = FastKrigingBuilder::<f64>::new()
            .points(points)
            .values(values)
            .covariance_function(CovarianceFunction::Exponential)
            .approximation_method(FastKrigingMethod::FixedRank(5))
            .build()
            .unwrap();

        let query_points = Array2::from_shape_vec(
            (3, 2),
            vec![0.1, 0.1, 0.5, 0.5, 0.9, 0.9],
        ).unwrap();

        let result = kriging.predict(&query_points.view()).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result.method, FastKrigingMethod::FixedRank(5));
        
        // Check that prediction completed without errors
        for &val in result.values().iter() {
            assert!(val.is_finite());
        }
    }

    #[cfg(feature = "linalg")]
    #[test]
    fn test_tapered_kriging() {
        let (points, values) = create_test_data(25, 2);
        
        let kriging = FastKrigingBuilder::<f64>::new()
            .points(points)
            .values(values)
            .covariance_function(CovarianceFunction::SquaredExponential)
            .approximation_method(FastKrigingMethod::Tapering(1.5))
            .build()
            .unwrap();

        let query_points = Array2::from_shape_vec(
            (2, 2),
            vec![0.3, 0.3, 0.7, 0.7],
        ).unwrap();

        let result = kriging.predict(&query_points.view()).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result.method, FastKrigingMethod::Tapering(1.5));
        
        // Check basic validity
        for &val in result.values().iter() {
            assert!(val.is_finite());
        }
        for &var in result.variances().iter() {
            assert!(var >= 0.0);
        }
    }

    #[cfg(feature = "linalg")]
    #[test]
    fn test_hodlr_kriging() {
        let (points, values) = create_test_data(40, 2);
        
        let kriging = FastKrigingBuilder::<f64>::new()
            .points(points)
            .values(values)
            .covariance_function(CovarianceFunction::Matern32)
            .approximation_method(FastKrigingMethod::HODLR(8))
            .build()
            .unwrap();

        let query_points = Array2::from_shape_vec(
            (2, 2),
            vec![0.4, 0.4, 0.6, 0.6],
        ).unwrap();

        let result = kriging.predict(&query_points.view()).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result.method, FastKrigingMethod::HODLR(8));
        
        // Basic sanity checks
        for &val in result.values().iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_predict_dimension_mismatch() {
        let (points, values) = create_test_data(10, 2);
        
        let kriging = FastKrigingBuilder::<f64>::new()
            .points(points)
            .values(values)
            .approximation_method(FastKrigingMethod::Local)
            .build()
            .unwrap();

        // Query points with wrong dimensionality
        let wrong_query = Array2::zeros((2, 3)); // 3D instead of 2D
        
        let result = kriging.predict(&wrong_query.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_query_prediction() {
        let (points, values) = create_test_data(10, 2);
        
        let kriging = FastKrigingBuilder::<f64>::new()
            .points(points)
            .values(values)
            .approximation_method(FastKrigingMethod::Local)
            .build()
            .unwrap();

        // Empty query points
        let empty_query = Array2::zeros((0, 2));
        
        let result = kriging.predict(&empty_query.view()).unwrap();
        assert_eq!(result.len(), 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_single_point_dataset() {
        let points = Array2::from_shape_vec((1, 2), vec![0.5, 0.5]).unwrap();
        let values = Array1::from_vec(vec![1.0]);
        
        let kriging = FastKrigingBuilder::<f64>::new()
            .points(points)
            .values(values)
            .approximation_method(FastKrigingMethod::Local)
            .max_neighbors(1)
            .build()
            .unwrap();

        let query_points = Array2::from_shape_vec((1, 2), vec![0.6, 0.6]).unwrap();
        
        // Should not crash on single point
        let result = kriging.predict(&query_points.view());
        assert!(result.is_ok());
    }
}
