//! API standardization guidelines and examples
//!
//! This module demonstrates the standardized API patterns that should be used
//! throughout the interpolation library for consistency.

use crate::traits::*;
use crate::{InterpolateError, InterpolateResult};
use scirs2_core::ndarray::{ArrayView1, ArrayView2};

/// Standard factory function pattern
///
/// All factory functions should follow this pattern:
/// 1. Name: `make_<interpolator_name>`
/// 2. Parameters: points, values, config (optional)
/// 3. Return: InterpolateResult<Interpolator>
///
/// # Example Implementation
/// ```ignore
/// pub fn make_example_interpolator<T: InterpolationFloat>(
///     points: &ArrayView2<T>,
///     values: &ArrayView1<T>,
///     config: Option<ExampleConfig>,
/// ) -> InterpolateResult<ExampleInterpolator<T>> {
///     // Validate input data
///     validation::validate_data_consistency(points, values)?;
///     
///     // Use default config if not provided
///     let config = config.unwrap_or_default();
///     config.validate()?;
///     
///     // Build and return interpolator
///     ExampleInterpolator::new(points, values, config)
/// }
/// ```
pub mod factory_pattern {
    use super::*;

    /// Example configuration structure
    #[derive(Debug, Clone)]
    pub struct StandardConfig<T: InterpolationFloat> {
        /// Smoothing parameter
        pub smoothing: Option<T>,

        /// Regularization parameter
        pub regularization: Option<T>,

        /// Maximum iterations for iterative methods
        pub max_iterations: usize,

        /// Convergence tolerance
        pub tolerance: T,
    }

    impl<T: InterpolationFloat> Default for StandardConfig<T> {
        fn default() -> Self {
            Self {
                smoothing: None,
                regularization: None,
                max_iterations: 100,
                tolerance: T::default_tolerance(),
            }
        }
    }

    impl<T: InterpolationFloat> InterpolationConfig for StandardConfig<T> {
        fn validate(&self) -> InterpolateResult<()> {
            if self.max_iterations == 0 {
                return Err(InterpolateError::invalid_input(
                    "max_iterations must be greater than 0",
                ));
            }

            if let Some(s) = self.smoothing {
                if s <= T::zero() {
                    return Err(InterpolateError::invalid_input(
                        "smoothing parameter must be positive",
                    ));
                }
            }

            Ok(())
        }

        fn default() -> Self {
            <Self as std::default::Default>::default()
        }
    }
}

/// Standard builder pattern
///
/// For more complex interpolators, use a builder pattern that follows
/// these conventions:
///
/// # Example
/// ```ignore
/// let interpolator = ExampleInterpolatorBuilder::new()
///     .with_smoothing(0.1)
///     .with_regularization(0.01)
///     .with_max_iterations(200)
///     .build(points, values)?;
/// ```
pub mod builder_pattern {
    use super::*;
    use crate::api_standards::factory_pattern::StandardConfig;

    /// Example builder structure
    #[derive(Debug, Clone)]
    pub struct StandardInterpolatorBuilder<T: InterpolationFloat> {
        config_factory_pattern: StandardConfig<T>,
    }

    impl<T: InterpolationFloat> StandardInterpolatorBuilder<T> {
        /// Create a new builder with default configuration
        pub fn new() -> Self {
            Self {
                config_factory_pattern: Default::default(),
            }
        }

        /// Set smoothing parameter
        pub fn with_smoothing(mut self, smoothing: T) -> Self {
            self.config_factory_pattern.smoothing = Some(smoothing);
            self
        }

        /// Set regularization parameter  
        pub fn with_regularization(mut self, regularization: T) -> Self {
            self.config_factory_pattern.regularization = Some(regularization);
            self
        }

        /// Set maximum iterations
        pub fn with_max_iterations(mut self, maxiterations: usize) -> Self {
            self.config_factory_pattern.max_iterations = maxiterations;
            self
        }

        /// Set convergence tolerance
        pub fn with_tolerance(mut self, tolerance: T) -> Self {
            self.config_factory_pattern.tolerance = tolerance;
            self
        }

        /// Build the interpolator
        pub fn build<I>(
            self,
            points: &ArrayView2<T>,
            values: &ArrayView1<T>,
        ) -> InterpolateResult<I>
        where
            for<'a> I: From<(
                ArrayView2<'a, T>,
                ArrayView1<'a, T>,
                factory_pattern::StandardConfig<T>,
            )>,
        {
            validation::validate_data_consistency(points, values)?;
            self.config_factory_pattern.validate()?;

            Ok(I::from((
                points.view(),
                values.view(),
                self.config_factory_pattern,
            )))
        }
    }

    impl<T: InterpolationFloat> Default for StandardInterpolatorBuilder<T> {
        fn default() -> Self {
            Self::new()
        }
    }
}

/// Standard evaluation interface
///
/// All interpolators should implement consistent evaluation methods
pub mod evaluation_pattern {
    use super::*;

    /// Standard batch evaluation with options
    pub fn evaluate_batch<T, I>(
        interpolator: &I,
        query_points: &ArrayView2<T>,
        options: Option<EvaluationOptions>,
    ) -> InterpolateResult<BatchEvaluationResult<T>>
    where
        T: InterpolationFloat,
        I: Interpolator<T>,
    {
        let _options = options.unwrap_or_default();

        // Validate query dimension
        // validation::validate_query_dimension(interpolator.data_dim(), query_points)?;

        // Perform evaluation
        let values = interpolator.evaluate(query_points)?;

        Ok(BatchEvaluationResult {
            values,
            uncertainties: None,
            out_of_bounds: Vec::new(),
        })
    }
}

/// Standard error handling
///
/// Consistent error creation and messages across the library
pub mod error_handling {
    use crate::InterpolateError;

    /// Create standard dimension mismatch error using structured error type
    pub fn dimension_mismatch(expected: usize, actual: usize, context: &str) -> InterpolateError {
        InterpolateError::dimension_mismatch(expected, actual, context)
    }

    /// Create standard empty data error using structured error type
    pub fn empty_data(context: &str) -> InterpolateError {
        InterpolateError::empty_data(context)
    }

    /// Create standard invalid parameter error using structured error type
    pub fn invalid_parameter<T: std::fmt::Display>(
        param: &str,
        expected: &str,
        actual: T,
        context: &str,
    ) -> InterpolateError {
        InterpolateError::invalid_parameter(param, expected, actual, context)
    }

    /// Create standard convergence failure error using structured error type
    pub fn convergence_failure(method: &str, iterations: usize) -> InterpolateError {
        InterpolateError::convergence_failure(method, iterations)
    }

    /// Create standard numerical instability error
    pub fn numerical_instability(context: &str, details: &str) -> InterpolateError {
        InterpolateError::numerical_instability(context, details)
    }

    /// Create standard insufficient points error
    pub fn insufficient_points(
        _required: usize,
        provided: usize,
        method: &str,
    ) -> InterpolateError {
        InterpolateError::insufficient_points(_required, provided, method)
    }
}

/// Standard input validation
///
/// Comprehensive validation utilities for consistent input checking
pub mod input_validation {
    use crate::{traits::InterpolationFloat, InterpolateError, InterpolateResult};
    use scirs2_core::ndarray::{ArrayView1, ArrayView2};

    /// Validate that data points are finite and well-formed
    pub fn validate_finite_data<T: InterpolationFloat>(
        points: &ArrayView2<T>,
        values: &ArrayView1<T>,
        context: &str,
    ) -> InterpolateResult<()> {
        // Check for NaN or infinite values in points
        for (i, point_slice) in points.outer_iter().enumerate() {
            for (j, &val) in point_slice.iter().enumerate() {
                if !val.is_finite() {
                    return Err(InterpolateError::InvalidInput {
                        message: format!(
                            "Non-finite value found in {context} points at position ({i}, {j}): {val}"
                        ),
                    });
                }
            }
        }

        // Check for NaN or infinite values in function values
        for (i, &val) in values.iter().enumerate() {
            if !val.is_finite() {
                return Err(InterpolateError::InvalidInput {
                    message: format!(
                        "Non-finite value found in {context} values at position {i}: {val}"
                    ),
                });
            }
        }

        Ok(())
    }

    /// Validate that data has sufficient points for the method
    pub fn validate_sufficient_points<T: InterpolationFloat>(
        points: &ArrayView2<T>,
        _values: &ArrayView1<T>,
        minimum_required: usize,
        method_name: &str,
    ) -> InterpolateResult<()> {
        let n_points = points.nrows();
        if n_points < minimum_required {
            return Err(InterpolateError::insufficient_points(
                minimum_required,
                n_points,
                method_name,
            ));
        }
        Ok(())
    }

    /// Validate query points have correct dimensions and are finite
    pub fn validate_query_points<T: InterpolationFloat>(
        query_points: &ArrayView2<T>,
        expected_dim: usize,
        context: &str,
    ) -> InterpolateResult<()> {
        if query_points.ncols() != expected_dim {
            return Err(InterpolateError::dimension_mismatch(
                expected_dim,
                query_points.ncols(),
                &format!("{context} query _points"),
            ));
        }

        // Check for finite values
        for (i, point_slice) in query_points.outer_iter().enumerate() {
            for (j, &val) in point_slice.iter().enumerate() {
                if !val.is_finite() {
                    return Err(InterpolateError::InvalidInput {
                        message: format!(
                            "Non-finite value found in {context} query _points at position ({i}, {j}): {val}"
                        ),
                    });
                }
            }
        }

        Ok(())
    }

    /// Validate parameter is positive
    pub fn validate_positive<T: InterpolationFloat>(
        value: T,
        param_name: &str,
        context: &str,
    ) -> InterpolateResult<()> {
        if value <= T::zero() {
            return Err(InterpolateError::invalid_parameter(
                param_name,
                "positive value",
                value,
                context,
            ));
        }
        Ok(())
    }

    /// Validate parameter is non-negative
    pub fn validate_non_negative<T: InterpolationFloat>(
        value: T,
        param_name: &str,
        context: &str,
    ) -> InterpolateResult<()> {
        if value < T::zero() {
            return Err(InterpolateError::invalid_parameter(
                param_name,
                "non-negative value",
                value,
                context,
            ));
        }
        Ok(())
    }

    /// Validate parameter is within a specific range
    pub fn validate_range<T: InterpolationFloat>(
        value: T,
        min: T,
        max: T,
        param_name: &str,
        context: &str,
    ) -> InterpolateResult<()> {
        if value < min || value > max {
            return Err(InterpolateError::invalid_parameter(
                param_name,
                format!("value between {min} and {max}"),
                value,
                context,
            ));
        }
        Ok(())
    }
}

/// Migration examples
///
/// Examples of how to migrate existing APIs to the new standard
pub mod migration_examples {
    use super::*;

    // Old API:
    // pub fn make_rbf_interpolator<F>(
    //     x: &ArrayView2<F>,
    //     y: &ArrayView1<F>,
    //     kernel: RBFKernel,
    //     epsilon: F,
    // ) -> InterpolateResult<RBFInterpolator<F>>

    // New standardized API:
    #[derive(Debug, Clone)]
    pub struct RBFConfig<T: InterpolationFloat> {
        pub kernel: RBFKernel,
        pub epsilon: T,
    }

    #[derive(Debug, Clone)]
    pub enum RBFKernel {
        Gaussian,
        Multiquadric,
        InverseMultiquadric,
        ThinPlate,
    }

    impl<T: InterpolationFloat> Default for RBFConfig<T> {
        fn default() -> Self {
            Self {
                kernel: RBFKernel::Gaussian,
                epsilon: T::from_f64(1.0).unwrap(),
            }
        }
    }

    impl<T: InterpolationFloat> InterpolationConfig for RBFConfig<T> {
        fn validate(&self) -> InterpolateResult<()> {
            if self.epsilon <= T::zero() {
                return Err(InterpolateError::invalid_input("epsilon must be positive"));
            }
            Ok(())
        }

        fn default() -> Self {
            <Self as std::default::Default>::default()
        }
    }

    /// Standardized RBF factory function
    pub fn make_rbf_interpolator<T: InterpolationFloat, I>(
        points: &ArrayView2<T>,
        values: &ArrayView1<T>,
        config: Option<RBFConfig<T>>,
    ) -> InterpolateResult<I>
    where
        for<'a> I: From<(ArrayView2<'a, T>, ArrayView1<'a, T>, RBFConfig<T>)>,
    {
        validation::validate_data_consistency(points, values)?;

        let config = config.unwrap_or_default();
        config.validate()?;

        Ok(I::from((points.view(), values.view(), config)))
    }
}
