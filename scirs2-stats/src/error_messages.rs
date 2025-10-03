//! Standardized error messages and error creation helpers
//!
//! This module provides consistent error messages and helper functions
//! for creating errors throughout the statistics module.

use crate::error::{StatsError, StatsResult};

/// Standard error messages for common validation failures
#[allow(dead_code)]
pub mod messages {
    /// Domain errors
    pub const POSITIVE_REQUIRED: &str = "Value must be positive (> 0)";
    pub const NON_NEGATIVE_REQUIRED: &str = "Value must be non-negative (>= 0)";
    pub const PROBABILITY_RANGE: &str = "Probability must be between 0 and 1 (inclusive)";
    pub const DEGREES_OF_FREEDOM_POSITIVE: &str = "Degrees of freedom must be positive";
    pub const SCALE_POSITIVE: &str = "Scale parameter must be positive";
    pub const SHAPE_POSITIVE: &str = "Shape parameter must be positive";
    pub const RATE_POSITIVE: &str = "Rate parameter must be positive";

    /// Dimension errors
    pub const ARRAYS_SAME_LENGTH: &str = "Input arrays must have the same length";
    pub const ARRAY_EMPTY: &str = "Input array cannot be empty";
    pub const MATRIX_SQUARE: &str = "Matrix must be square";
    pub const INSUFFICIENT_DATA: &str = "Insufficient data for calculation";

    /// Computation errors
    pub const NUMERICAL_OVERFLOW: &str = "Numerical overflow occurred during computation";
    pub const CONVERGENCE_FAILED: &str = "Algorithm failed to converge";
    pub const SINGULAR_MATRIX: &str = "Matrix is singular or near-singular";

    /// Not implemented
    pub const FEATURE_NOT_IMPLEMENTED: &str = "This feature is not yet implemented";
}

/// Helper functions for creating standardized errors
#[allow(dead_code)]
pub mod helpers {
    use super::*;

    /// Create a domain error for a parameter that must be positive
    pub fn positive_required(_paramname: &str, value: impl std::fmt::Display) -> StatsError {
        StatsError::domain(format!("{_paramname} must be positive (> 0), got {value}"))
    }

    /// Create a domain error for a parameter that must be non-negative
    pub fn non_negative_required(_paramname: &str, value: impl std::fmt::Display) -> StatsError {
        StatsError::domain(format!(
            "{_paramname} must be non-negative (>= 0), got {value}"
        ))
    }

    /// Create a domain error for a probability parameter
    pub fn probability_range(_paramname: &str, value: impl std::fmt::Display) -> StatsError {
        StatsError::domain(format!(
            "{_paramname} must be between 0 and 1 (inclusive), got {value}"
        ))
    }

    /// Create a dimension mismatch error for arrays that should have the same length
    pub fn arrays_length_mismatch(len1: usize, len2: usize) -> StatsError {
        StatsError::dimension_mismatch(format!(
            "Arrays must have the same length, got {len1} and {len2}"
        ))
    }

    /// Create an invalid argument error for empty arrays
    pub fn array_empty(_arrayname: &str) -> StatsError {
        StatsError::invalid_argument(format!("{_arrayname} cannot be empty"))
    }

    /// Create an invalid argument error for insufficient data
    pub fn insufficientdata(required: usize, actual: usize, context: &str) -> StatsError {
        StatsError::invalid_argument(format!(
            "Insufficient data for {context}: requires at least {required} samples, got {actual}"
        ))
    }

    /// Create a computation error for numerical issues
    pub fn numerical_error(context: &str) -> StatsError {
        StatsError::computation(format!(
            "Numerical error in {context}: check for extreme values or scaling issues"
        ))
    }

    /// Create a not implemented error with feature name
    pub fn not_implemented(feature: &str) -> StatsError {
        StatsError::not_implemented(format!("{feature} is not yet implemented"))
    }
}

/// Validation helpers that return standardized errors
#[allow(dead_code)]
pub mod validation {
    use super::*;
    use scirs2_core::numeric::Float;

    /// Validate that a value is positive
    pub fn ensure_positive<F: Float + std::fmt::Display>(
        value: F,
        param_name: &str,
    ) -> StatsResult<F> {
        if value <= F::zero() {
            Err(helpers::positive_required(param_name, value))
        } else {
            Ok(value)
        }
    }

    /// Validate that a value is non-negative
    pub fn ensure_non_negative<F: Float + std::fmt::Display>(
        value: F,
        param_name: &str,
    ) -> StatsResult<F> {
        if value < F::zero() {
            Err(helpers::non_negative_required(param_name, value))
        } else {
            Ok(value)
        }
    }

    /// Validate that a value is a valid probability [0, 1]
    pub fn ensure_probability<F: Float + std::fmt::Display>(
        value: F,
        param_name: &str,
    ) -> StatsResult<F> {
        if value < F::zero() || value > F::one() {
            Err(helpers::probability_range(param_name, value))
        } else {
            Ok(value)
        }
    }

    /// Validate that arrays have the same length
    pub fn ensure_same_length<T, U>(arr1: &[T], arr2: &[U]) -> StatsResult<()> {
        if arr1.len() != arr2.len() {
            Err(helpers::arrays_length_mismatch(arr1.len(), arr2.len()))
        } else {
            Ok(())
        }
    }

    /// Validate that an array is not empty
    pub fn ensure_not_empty<T>(_arr: &[T], arrayname: &str) -> StatsResult<()> {
        if _arr.is_empty() {
            Err(helpers::array_empty(arrayname))
        } else {
            Ok(())
        }
    }

    /// Validate that we have sufficient data
    pub fn ensure_sufficientdata(actual: usize, required: usize, context: &str) -> StatsResult<()> {
        if actual < required {
            Err(helpers::insufficientdata(required, actual, context))
        } else {
            Ok(())
        }
    }
}
