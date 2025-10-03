//! Enhanced error context and recovery suggestions
//!
//! This module provides enhanced error context with detailed recovery suggestions
//! for common statistical computation errors.

use crate::error::{StatsError, StatsResult};
use std::fmt::Display;

/// Error context with detailed information and recovery suggestions
#[derive(Debug)]
pub struct EnhancedError {
    /// The original error
    pub error: StatsError,
    /// Additional context about where the error occurred
    pub context: String,
    /// Specific recovery suggestions
    pub suggestions: Vec<String>,
    /// Related documentation or examples
    pub see_also: Vec<String>,
}

impl EnhancedError {
    /// Create a new enhanced error
    pub fn new(error: StatsError, context: impl Into<String>) -> Self {
        Self {
            error,
            context: context.into(),
            suggestions: Vec::new(),
            see_also: Vec::new(),
        }
    }

    /// Add a recovery suggestion
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestions.push(suggestion.into());
        self
    }

    /// Add multiple recovery suggestions
    pub fn with_suggestions(
        mut self,
        suggestions: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.suggestions
            .extend(suggestions.into_iter().map(|s| s.into()));
        self
    }

    /// Add a reference to related documentation or examples
    pub fn see_also(mut self, reference: impl Into<String>) -> Self {
        self.see_also.push(reference.into());
        self
    }

    /// Convert to StatsError with formatted message
    pub fn into_error(self) -> StatsError {
        let mut message = format!("{}\nContext: {}", self.error, self.context);

        if !self.suggestions.is_empty() {
            message.push_str("\n\nSuggestions:");
            for (i, suggestion) in self.suggestions.iter().enumerate() {
                message.push_str(&format!("\n  {}. {}", i + 1, suggestion));
            }
        }

        if !self.see_also.is_empty() {
            message.push_str("\n\nSee also:");
            for reference in &self.see_also {
                message.push_str(&format!("\n  - {}", reference));
            }
        }

        StatsError::computation(message)
    }
}

/// Enhanced validation with detailed error context
pub mod enhanced_validation {
    use super::*;
    use scirs2_core::numeric::Float;

    /// Validate distribution parameters with enhanced error messages
    pub fn validate_distribution_params<F: Float + Display>(
        params: &[(F, &str, ParamType)],
        distribution_name: &str,
    ) -> StatsResult<()> {
        for &(value, name, param_type) in params {
            match param_type {
                ParamType::Positive => {
                    if value <= F::zero() {
                        return Err(EnhancedError::new(
                            StatsError::domain(format!("{} must be positive, got {}", name, value)),
                            format!("Invalid {} parameter for {} distribution", name, distribution_name),
                        )
                        .with_suggestions(vec![
                            format!("Ensure {} > 0", name),
                            "Check your data preprocessing steps".to_string(),
                            "Consider using a different distribution if negative values are expected".to_string(),
                        ])
                        .see_also(format!("distributions::{}", distribution_name.to_lowercase()))
                        .into_error());
                    }
                }
                ParamType::NonNegative => {
                    if value < F::zero() {
                        return Err(EnhancedError::new(
                            StatsError::domain(format!(
                                "{} must be non-negative, got {}",
                                name, value
                            )),
                            format!(
                                "Invalid {} parameter for {} distribution",
                                name, distribution_name
                            ),
                        )
                        .with_suggestions(vec![
                            format!("Ensure {} >= 0", name),
                            "Check for data entry errors".to_string(),
                        ])
                        .into_error());
                    }
                }
                ParamType::Probability => {
                    if value < F::zero() || value > F::one() {
                        return Err(EnhancedError::new(
                            StatsError::domain(format!(
                                "{} must be in [0, 1], got {}",
                                name, value
                            )),
                            format!(
                                "Invalid probability parameter '{}' for {} distribution",
                                name, distribution_name
                            ),
                        )
                        .with_suggestions(vec![
                            "Ensure probability is between 0 and 1 (inclusive)",
                            "Check if you're using a proportion instead of a percentage",
                            "Verify your probability calculations",
                        ])
                        .into_error());
                    }
                }
                ParamType::Integer => {
                    if value.floor() != value {
                        return Err(EnhancedError::new(
                            StatsError::domain(format!(
                                "{} must be an integer, got {}",
                                name, value
                            )),
                            format!(
                                "Invalid {} parameter for {} distribution",
                                name, distribution_name
                            ),
                        )
                        .with_suggestions(vec![
                            "Round to the nearest integer if appropriate",
                            "Check if you're using the correct distribution",
                        ])
                        .into_error());
                    }
                }
                ParamType::PositiveInteger => {
                    if value.floor() != value || value <= F::zero() {
                        return Err(EnhancedError::new(
                            StatsError::domain(format!(
                                "{} must be a positive integer, got {}",
                                name, value
                            )),
                            format!(
                                "Invalid {} parameter for {} distribution",
                                name, distribution_name
                            ),
                        )
                        .with_suggestions(vec![
                            "Ensure the value is a positive whole number",
                            "Check your counting or indexing logic",
                        ])
                        .into_error());
                    }
                }
            }
        }
        Ok(())
    }

    /// Parameter type for validation
    #[derive(Debug, Copy, Clone)]
    pub enum ParamType {
        Positive,
        NonNegative,
        Probability,
        Integer,
        PositiveInteger,
    }
}

/// Enhanced error handling for numerical computations
pub mod numerical {
    use super::*;

    /// Handle numerical overflow with context
    pub fn handle_overflow(operation: &str, values: &[impl Display]) -> StatsError {
        let value_str = values
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ");

        EnhancedError::new(
            StatsError::computation("Numerical overflow"),
            format!(
                "Overflow occurred during {} with values: [{}]",
                operation, value_str
            ),
        )
        .with_suggestions(vec![
            "Scale your input data to smaller magnitudes",
            "Use logarithmic transformations if appropriate",
            "Consider using higher precision data types (f64 instead of f32)",
            "Check for extreme outliers in your data",
        ])
        .see_also("numerical_stability")
        .into_error()
    }

    /// Handle convergence failure with context
    pub fn handle_convergence_failure(
        algorithm: &str,
        iterations: usize,
        tolerance: f64,
    ) -> StatsError {
        EnhancedError::new(
            StatsError::computation("Algorithm failed to converge"),
            format!(
                "{} failed to converge after {} iterations (tolerance: {})",
                algorithm, iterations, tolerance
            ),
        )
        .with_suggestions(vec![
            "Increase the maximum number of iterations",
            "Relax the convergence tolerance",
            "Check if your data is well-conditioned",
            "Try different initial values",
            "Consider using a different algorithm",
        ])
        .into_error()
    }

    /// Handle singular matrix errors
    pub fn handle_singular_matrix(context: &str) -> StatsError {
        EnhancedError::new(
            StatsError::computation("Matrix is singular or near-singular"),
            format!("Singular matrix encountered in {}", context),
        )
        .with_suggestions(vec![
            "Check for linear dependencies in your data",
            "Remove collinear features",
            "Add regularization to your model",
            "Ensure you have more observations than features",
            "Check for duplicate rows or columns",
        ])
        .see_also("linear_algebra")
        .into_error()
    }
}

/// Enhanced error handling for data validation
pub mod data_validation {
    use super::*;
    use scirs2_core::numeric::Float;

    /// Validate input data with enhanced error messages
    pub fn validatedata_quality<T>(data: &[T], context: &str, allow_empty: bool) -> StatsResult<()>
    where
        T: Float + Display,
    {
        if data.is_empty() && !allow_empty {
            return Err(EnhancedError::new(
                StatsError::invalid_argument("Empty data array"),
                format!("Empty input data for {}", context),
            )
            .with_suggestions(vec![
                "Ensure your data loading process completed successfully",
                "Check if filters removed all data points",
                "Verify the data source is not _empty",
            ])
            .into_error());
        }

        // Check for NaN or infinite values
        let nan_count = data.iter().filter(|&&x| x.is_nan()).count();
        let inf_count = data.iter().filter(|&&x| x.is_infinite()).count();

        if nan_count > 0 {
            return Err(EnhancedError::new(
                StatsError::invalid_argument(format!("Found {} NaN values", nan_count)),
                format!("Invalid data values in {}", context),
            )
            .with_suggestions(vec![
                "Use dropna() or similar to remove NaN values",
                "Check for division by zero in calculations",
                "Verify data import didn't introduce NaN values",
                "Consider imputation methods if appropriate",
            ])
            .see_also("data_preprocessing")
            .into_error());
        }

        if inf_count > 0 {
            return Err(EnhancedError::new(
                StatsError::invalid_argument(format!("Found {} infinite values", inf_count)),
                format!("Invalid data values in {}", context),
            )
            .with_suggestions(vec![
                "Check for numerical overflow in calculations",
                "Apply bounds checking before operations",
                "Consider log transformations for large values",
                "Remove or cap extreme outliers",
            ])
            .into_error());
        }

        Ok(())
    }

    /// Validate array shapes match requirements
    pub fn validateshape_compatibility(
        actualshape: &[usize],
        expectedshape: &[Option<usize>],
        array_name: &str,
    ) -> StatsResult<()> {
        if actualshape.len() != expectedshape.len() {
            return Err(EnhancedError::new(
                StatsError::dimension_mismatch(format!(
                    "Expected {}-dimensional array, got {}-dimensional",
                    expectedshape.len(),
                    actualshape.len()
                )),
                format!("Shape mismatch for {}", array_name),
            )
            .with_suggestions(vec![
                "Reshape your array using reshape() or similar",
                "Check if you're passing the correct variable",
                "Verify data preprocessing steps maintained correct dimensions",
            ])
            .into_error());
        }

        for (i, (&actual, &expected)) in actualshape.iter().zip(expectedshape.iter()).enumerate() {
            if let Some(expected_dim) = expected {
                if actual != expected_dim {
                    return Err(EnhancedError::new(
                        StatsError::dimension_mismatch(format!(
                            "Dimension {} mismatch: expected {}, got {}",
                            i, expected_dim, actual
                        )),
                        format!("Invalid shape for {}", array_name),
                    )
                    .with_suggestions(vec![
                        format!("Ensure dimension {} has size {}", i, expected_dim),
                        "Check array slicing or indexing operations".to_string(),
                    ])
                    .into_error());
                }
            }
        }

        Ok(())
    }
}
