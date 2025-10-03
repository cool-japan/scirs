//! Common traits for interpolation types
//!
//! This module defines standard trait bounds used throughout the interpolation library
//! to ensure API consistency and reduce repetition.

use scirs2_core::ndarray::{ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::{Debug, Display, LowerExp};
use std::ops::{AddAssign, DivAssign, MulAssign, RemAssign, SubAssign};

/// Type alias for confidence intervals result
pub type ConfidenceIntervals<T> = Vec<(T, Vec<(T, T)>)>;

/// Standard floating-point type for interpolation operations
///
/// This trait combines all the common bounds needed for interpolation algorithms,
/// providing a single consistent constraint across the library.
pub trait InterpolationFloat:
    Float
    + FromPrimitive
    + Debug
    + Display
    + LowerExp
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + RemAssign
    + Send
    + Sync
    + Default
    + Copy
    + std::iter::Sum
    + 'static
{
    /// Default epsilon value for this floating-point type
    fn default_epsilon() -> Self {
        Self::from_f64(1e-9).unwrap_or_else(|| Self::epsilon())
    }

    /// Default tolerance for iterative algorithms
    fn default_tolerance() -> Self {
        Self::from_f64(1e-12).unwrap_or_else(|| Self::epsilon() * Self::from_f64(100.0).unwrap())
    }
}

// Implement for standard floating-point types
impl InterpolationFloat for f32 {}
impl InterpolationFloat for f64 {}

/// Standard input data format for interpolation
///
/// This trait defines the expected format for input data points and values
pub trait InterpolationData<T: InterpolationFloat> {
    /// Get the spatial coordinates of the data points
    fn points(&self) -> ArrayView2<T>;

    /// Get the function values at the data points
    fn values(&self) -> ArrayView1<T>;

    /// Get the number of data points
    fn len(&self) -> usize {
        self.values().len()
    }

    /// Check if the data is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the spatial dimension of the data
    fn dim(&self) -> usize {
        self.points().ncols()
    }
}

/// Standard query interface for interpolators
pub trait Interpolator<T: InterpolationFloat> {
    /// Evaluate the interpolator at given query points
    fn evaluate(&self, querypoints: &ArrayView2<T>) -> crate::InterpolateResult<Vec<T>>;

    /// Evaluate the interpolator at a single point
    fn evaluate_single(&self, point: &ArrayView1<T>) -> crate::InterpolateResult<T> {
        let query = point.view().insert_axis(scirs2_core::ndarray::Axis(0));
        self.evaluate(&query).map(|v| v[0])
    }

    /// Evaluate derivatives at query points (if supported)
    fn evaluate_derivatives(
        &self,
        query_points: &ArrayView2<T>,
        order: usize,
    ) -> crate::InterpolateResult<Vec<Vec<T>>> {
        let _ = (query_points, order);
        Err(crate::InterpolateError::NotImplemented(
            "Derivative evaluation not implemented for this interpolator".to_string(),
        ))
    }

    /// Evaluate with options for advanced control
    fn evaluate_with_options(
        &self,
        query_points: &ArrayView2<T>,
        options: &EvaluationOptions,
    ) -> crate::InterpolateResult<BatchEvaluationResult<T>> {
        let _ = options;
        let values = self.evaluate(query_points)?;
        Ok(BatchEvaluationResult {
            values,
            uncertainties: None,
            out_of_bounds: Vec::new(),
        })
    }

    /// Get the spatial dimension of the interpolator
    fn dimension(&self) -> usize;

    /// Get the number of data points used to construct this interpolator
    fn len(&self) -> usize;

    /// Check if the interpolator is empty (no data points)
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Advanced interpolator interface for methods with uncertainty quantification
pub trait UncertaintyInterpolator<T: InterpolationFloat>: Interpolator<T> {
    /// Evaluate with uncertainty estimates
    fn evaluate_with_uncertainty(
        &self,
        query_points: &ArrayView2<T>,
    ) -> crate::InterpolateResult<(Vec<T>, Vec<T>)>;

    /// Evaluate confidence intervals at specified levels
    fn evaluate_confidence_intervals(
        &self,
        query_points: &ArrayView2<T>,
        confidence_levels: &[T],
    ) -> crate::InterpolateResult<ConfidenceIntervals<T>>;
}

/// Adaptive interpolator interface for methods that can refine their approximation
pub trait AdaptiveInterpolator<T: InterpolationFloat>: Interpolator<T> {
    /// Add new data _points to refine the interpolation
    fn add_points(
        &mut self,
        new_points: &ArrayView2<T>,
        new_values: &ArrayView1<T>,
    ) -> crate::InterpolateResult<()>;

    /// Remove data _points from the interpolation
    fn remove_points(&mut self, indices: &[usize]) -> crate::InterpolateResult<()>;

    /// Update the interpolation with new data
    fn update(
        &mut self,
        points: &ArrayView2<T>,
        values: &ArrayView1<T>,
    ) -> crate::InterpolateResult<()>;
}

/// Configuration trait for interpolation methods
pub trait InterpolationConfig: Clone + Debug {
    /// Validate the configuration
    fn validate(&self) -> crate::InterpolateResult<()>;

    /// Get default configuration
    fn default() -> Self;
}

/// Builder pattern trait for consistent API
pub trait InterpolatorBuilder<T: InterpolationFloat> {
    /// The interpolator type this builder creates
    type Interpolator: Interpolator<T>;

    /// The configuration type for this builder
    type Config: InterpolationConfig;

    /// Build the interpolator with the given data and configuration
    fn build(
        points: &ArrayView2<T>,
        values: &ArrayView1<T>,
        config: Self::Config,
    ) -> crate::InterpolateResult<Self::Interpolator>;
}

/// Standard evaluation options
#[derive(Debug, Clone)]
pub struct EvaluationOptions {
    /// Number of parallel workers (None for automatic)
    pub workers: Option<usize>,

    /// Whether to use caching for repeated evaluations
    pub use_cache: bool,

    /// Whether to validate input bounds
    pub validate_bounds: bool,

    /// Fill value for out-of-bounds queries
    pub fill_value: Option<f64>,
}

impl Default for EvaluationOptions {
    fn default() -> Self {
        Self {
            workers: None,
            use_cache: false,
            validate_bounds: true,
            fill_value: None,
        }
    }
}

/// Standard result for batch evaluation
pub struct BatchEvaluationResult<T: InterpolationFloat> {
    /// The interpolated values
    pub values: Vec<T>,

    /// Optional uncertainty estimates (for methods that support it)
    pub uncertainties: Option<Vec<T>>,

    /// Indices of out-of-bounds points (if any)
    pub out_of_bounds: Vec<usize>,
}

/// Spline-specific interface for methods that support derivatives and integrals
pub trait SplineInterpolator<T: InterpolationFloat>: Interpolator<T> {
    /// Evaluate the nth derivative at query points
    fn derivative(
        &self,
        query_points: &ArrayView2<T>,
        order: usize,
    ) -> crate::InterpolateResult<Vec<T>>;

    /// Evaluate the definite integral over specified bounds
    fn integrate(&self, bounds: &[(T, T)]) -> crate::InterpolateResult<Vec<T>>;

    /// Get the antiderivative as a new spline
    fn antiderivative(&self) -> crate::InterpolateResult<Box<dyn SplineInterpolator<T>>>;

    /// Find roots of the spline within given bounds
    fn find_roots(&self, bounds: &[(T, T)], tolerance: T) -> crate::InterpolateResult<Vec<T>>;

    /// Find extrema (local minima and maxima) within given bounds
    fn find_extrema(
        &self,
        bounds: &[(T, T)],
        tolerance: T,
    ) -> crate::InterpolateResult<Vec<(T, T, ExtremaType)>>;
}

/// Type of extrema for spline analysis
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExtremaType {
    /// Local minimum
    Minimum,
    /// Local maximum
    Maximum,
    /// Inflection point
    InflectionPoint,
}

/// Performance monitoring interface for interpolation methods
pub trait PerformanceMonitoring {
    /// Get timing statistics for the last operation
    fn get_timing_stats(&self) -> Option<TimingStats>;

    /// Get memory usage statistics
    fn get_memory_stats(&self) -> Option<MemoryStats>;

    /// Enable or disable performance monitoring
    fn set_monitoring_enabled(&mut self, enabled: bool);

    /// Reset performance counters
    fn reset_stats(&mut self);
}

/// Timing statistics for performance monitoring
#[derive(Debug, Clone)]
pub struct TimingStats {
    /// Time spent in construction (microseconds)
    pub construction_time_us: u64,
    /// Time spent in evaluation (microseconds)
    pub evaluation_time_us: u64,
    /// Number of evaluations performed
    pub evaluation_count: u64,
    /// Average time per evaluation (microseconds)
    pub avg_evaluation_time_us: f64,
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Memory used by the interpolator (bytes)
    pub interpolator_memory_bytes: usize,
    /// Memory used by cached data (bytes)  
    pub cache_memory_bytes: usize,
    /// Total allocations
    pub total_allocations: usize,
    /// Peak memory usage (bytes)
    pub peak_memory_bytes: usize,
}

/// Serialization support for interpolators
pub trait SerializableInterpolator<T: InterpolationFloat> {
    /// Serialize the interpolator to bytes
    fn serialize(&self) -> crate::InterpolateResult<Vec<u8>>;

    /// Deserialize the interpolator from bytes
    fn deserialize(data: &[u8]) -> crate::InterpolateResult<Self>
    where
        Self: Sized;

    /// Get the serialization format version
    fn serialization_version(&self) -> u32;
}

/// Parallel evaluation support for interpolators
pub trait ParallelInterpolator<T: InterpolationFloat>: Interpolator<T> + Sync + Send {
    /// Evaluate at query points using parallel execution
    fn evaluate_parallel(
        &self,
        query_points: &ArrayView2<T>,
        num_threads: Option<usize>,
    ) -> crate::InterpolateResult<Vec<T>>;

    /// Check if this interpolator supports parallel evaluation
    fn supports_parallel(&self) -> bool {
        true
    }

    /// Get the recommended number of threads for this interpolator
    fn recommended_thread_count(&self, query_size: usize) -> usize {
        (query_size / 1000).max(1).min(num_cpus::get())
    }
}

/// Common validation utilities
pub mod validation {
    use super::*;

    /// Validate that points and values have consistent dimensions
    pub fn validate_data_consistency<T: InterpolationFloat>(
        points: &ArrayView2<T>,
        values: &ArrayView1<T>,
    ) -> crate::InterpolateResult<()> {
        if points.nrows() != values.len() {
            return Err(crate::InterpolateError::invalid_input(format!(
                "Inconsistent data dimensions: {} points but {} values",
                points.nrows(),
                values.len()
            )));
        }

        if points.is_empty() || values.is_empty() {
            return Err(crate::InterpolateError::invalid_input("Empty input data"));
        }

        Ok(())
    }

    /// Validate query points have correct dimension
    pub fn validate_query_dimension<T: InterpolationFloat>(
        data_dim: usize,
        query_points: &ArrayView2<T>,
    ) -> crate::InterpolateResult<()> {
        if query_points.ncols() != data_dim {
            return Err(crate::InterpolateError::invalid_input(format!(
                "Query dimension {} does not match data dimension {}",
                query_points.ncols(),
                data_dim
            )));
        }
        Ok(())
    }

    /// Validate that query points are within valid bounds (if bounds checking is enabled)
    pub fn validate_query_bounds<T: InterpolationFloat>(
        query_points: &ArrayView2<T>,
        bounds: &[(T, T)],
        options: &EvaluationOptions,
    ) -> crate::InterpolateResult<Vec<usize>> {
        if !options.validate_bounds {
            return Ok(Vec::new());
        }

        let mut out_of_bounds = Vec::new();

        for (row_idx, point) in query_points.outer_iter().enumerate() {
            for (&coord, &(min_bound, max_bound)) in point.iter().zip(bounds.iter()) {
                if coord < min_bound || coord > max_bound {
                    out_of_bounds.push(row_idx);
                    break;
                }
            }
        }

        Ok(out_of_bounds)
    }

    /// Validate interpolator configuration parameters
    pub fn validate_config<C: InterpolationConfig>(config: &C) -> crate::InterpolateResult<()> {
        config.validate()
    }
}
