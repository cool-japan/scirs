//! Error types for the SciRS2 interpolation module

use thiserror::Error;

/// Interpolation error type with specific context
#[derive(Error, Debug)]
pub enum InterpolateError {
    /// Invalid input data with specific details
    #[error("Invalid input data: {message}")]
    InvalidInput { message: String },

    /// Domain error with specific point and bounds
    #[error("Point {point:?} is outside domain [{min}, {max}] in {context}")]
    OutOfDomain {
        point: String,
        min: String,
        max: String,
        context: String,
    },

    /// Invalid parameter value with expected range
    #[error("Invalid {parameter}: expected {expected}, got {actual} in {context}")]
    InvalidParameter {
        parameter: String,
        expected: String,
        actual: String,
        context: String,
    },

    /// Shape mismatch with specific details
    #[error("Shape mismatch: expected {expected}, got {actual} for {object}")]
    ShapeMismatch {
        expected: String,
        actual: String,
        object: String,
    },

    /// Computation error (generic error)
    #[error("Computation error: {0}")]
    ComputationError(String),

    /// Shape error (ndarray shape mismatch)
    #[error("Shape error: {0}")]
    ShapeError(String),

    /// Not implemented error
    #[error("Not implemented: {0}")]
    NotImplemented(String),

    /// Invalid value provided
    #[error("Invalid value: {0}")]
    InvalidValue(String),

    /// Dimension mismatch between arrays
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// Point is outside the interpolation range
    #[error("Out of bounds: {0}")]
    OutOfBounds(String),

    /// Invalid interpolator state
    #[error("Invalid state: {0}")]
    InvalidState(String),

    /// Invalid operation attempted
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    /// Special case for boundary handling: point was mapped to an equivalent point
    /// This is a special case that's not really an error, but used for control flow
    #[error("Point was mapped to {0}")]
    MappedPoint(f64),

    /// Generic version of MappedPoint that can handle any numeric type
    /// Used for control flow in generic interpolation functions
    #[error("Point was mapped to equivalent")]
    MappedPointGeneric(Box<dyn std::any::Any + Send + Sync>),

    /// Index out of bounds error
    #[error("Index error: {0}")]
    IndexError(String),

    /// I/O error
    #[error("IO error: {0}")]
    IoError(String),

    /// Linear algebra error
    #[error("Linear algebra error: {0}")]
    LinalgError(String),

    /// Numerical error (e.g., division by zero, overflow)
    #[error("Numerical error: {0}")]
    NumericalError(String),

    /// Operation is not supported
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    /// Insufficient data for the operation
    #[error("Insufficient data: {0}")]
    InsufficientData(String),

    /// Interpolation failed
    #[error("Interpolation failed: {0}")]
    InterpolationFailed(String),

    /// Missing points data
    #[error("Missing points data: interpolator requires training points")]
    MissingPoints,

    /// Missing values data
    #[error("Missing values data: interpolator requires training values")]
    MissingValues,

    /// Numerical instability detected
    #[error("Numerical instability: {message}")]
    NumericalInstability { message: String },
}

impl From<scirs2_core::ndarray::ShapeError> for InterpolateError {
    fn from(err: scirs2_core::ndarray::ShapeError) -> Self {
        InterpolateError::ShapeError(err.to_string())
    }
}

impl From<scirs2_core::CoreError> for InterpolateError {
    fn from(err: scirs2_core::CoreError) -> Self {
        InterpolateError::ComputationError(err.to_string())
    }
}

/// Result type for interpolation operations
pub type InterpolateResult<T> = Result<T, InterpolateError>;

impl InterpolateError {
    /// Create an InvalidInput error with a descriptive message
    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::InvalidInput {
            message: message.into(),
        }
    }

    /// Create an OutOfDomain error with specific context
    pub fn out_of_domain<T: std::fmt::Display>(
        point: T,
        min: T,
        max: T,
        context: impl Into<String>,
    ) -> Self {
        Self::OutOfDomain {
            point: point.to_string(),
            min: min.to_string(),
            max: max.to_string(),
            context: context.into(),
        }
    }

    /// Create an InvalidParameter error with specific context  
    pub fn invalid_parameter<T: std::fmt::Display>(
        parameter: impl Into<String>,
        expected: impl Into<String>,
        actual: T,
        context: impl Into<String>,
    ) -> Self {
        Self::InvalidParameter {
            parameter: parameter.into(),
            expected: expected.into(),
            actual: actual.to_string(),
            context: context.into(),
        }
    }

    /// Create a ShapeMismatch error with specific details
    pub fn shape_mismatch(
        expected: impl Into<String>,
        actual: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self::ShapeMismatch {
            expected: expected.into(),
            actual: actual.into(),
            object: object.into(),
        }
    }

    /// Create a standard dimension mismatch error
    pub fn dimension_mismatch(expected: usize, actual: usize, context: &str) -> Self {
        Self::DimensionMismatch(format!(
            "Dimension mismatch in {context}: expected {expected}, got {actual}"
        ))
    }

    /// Create a standard empty data error
    pub fn empty_data(context: &str) -> Self {
        Self::InsufficientData(format!("Empty input data provided to {context}"))
    }

    /// Create a standard convergence failure error
    pub fn convergence_failure(method: &str, iterations: usize) -> Self {
        Self::ComputationError(format!(
            "{method} failed to converge after {iterations} iterations"
        ))
    }

    /// Create a numerical stability error
    pub fn numerical_instability(context: &str, details: &str) -> Self {
        Self::NumericalError(format!("Numerical instability in {context}: {details}"))
    }

    /// Create a numerical error
    pub fn numerical_error(message: impl Into<String>) -> Self {
        Self::NumericalError(message.into())
    }

    /// Create an insufficient data points error
    pub fn insufficient_points(required: usize, provided: usize, method: &str) -> Self {
        Self::InsufficientData(format!(
            "{method} requires at least {required} points, but only {provided} provided"
        ))
    }

    /// Create an actionable parameter error with suggestions
    pub fn invalid_parameter_with_suggestion<T: std::fmt::Display>(
        parameter: impl Into<String>,
        value: T,
        context: impl Into<String>,
        suggestion: impl Into<String>,
    ) -> Self {
        Self::InvalidParameter {
            parameter: parameter.into(),
            expected: suggestion.into(),
            actual: value.to_string(),
            context: context.into(),
        }
    }

    /// Create an actionable domain error with recovery suggestions
    pub fn out_of_domain_with_suggestion<T: std::fmt::Display>(
        point: T,
        min: T,
        max: T,
        context: impl Into<String>,
        suggestion: impl Into<String>,
    ) -> Self {
        let context_str = context.into();
        let suggestion_str = suggestion.into();
        Self::OutOfDomain {
            point: point.to_string(),
            min: min.to_string(),
            max: max.to_string(),
            context: format!("{context_str} - Suggestion: {suggestion_str}"),
        }
    }

    /// Create a numerical stability error with actionable advice
    pub fn numerical_instability_with_advice(context: &str, details: &str, advice: &str) -> Self {
        Self::NumericalError(format!(
            "Numerical instability in {context}: {details} - ADVICE: {advice}"
        ))
    }

    /// Create a convergence failure with actionable recommendations
    pub fn convergence_failure_with_advice(method: &str, iterations: usize, advice: &str) -> Self {
        Self::ComputationError(format!(
            "{method} failed to converge after {iterations} iterations - RECOMMENDATION: {advice}"
        ))
    }

    /// Create a matrix conditioning error with specific recommendations
    pub fn matrix_conditioning_error(
        condition_number: f64,
        context: &str,
        recommended_regularization: Option<f64>,
    ) -> Self {
        let advice = if let Some(reg) = recommended_regularization {
            format!(
                "Matrix is ill-conditioned (condition number: {condition_number:.2e}). Try regularization parameter ≥ {reg:.2e}"
            )
        } else {
            format!(
                "Matrix is ill-conditioned (condition number: {condition_number:.2e}). Consider data preprocessing or regularization"
            )
        };

        Self::LinalgError(format!(
            "{context}: {} - SOLUTION: {advice}",
            if condition_number > 1e16 {
                "Severe numerical instability"
            } else {
                "Poor numerical conditioning"
            }
        ))
    }

    /// Create a data quality error with preprocessing suggestions
    pub fn data_quality_error(issue: &str, context: &str, preprocessingadvice: &str) -> Self {
        Self::InvalidInput {
            message: format!(
                "{issue} detected in {context}: Data may be unsuitable for interpolation - DATA PREPROCESSING: {preprocessingadvice}"
            ),
        }
    }

    /// Create a method selection error with alternative suggestions
    pub fn method_selection_error(
        attempted_method: &str,
        data_characteristics: &str,
        recommended_alternatives: &[&str],
    ) -> Self {
        let alternatives = recommended_alternatives.join(", ");
        Self::UnsupportedOperation(format!(
            "{attempted_method} is not suitable for data with {data_characteristics}: Consider using a different interpolation method - ALTERNATIVES: Try {alternatives}"
        ))
    }

    /// Create a performance warning with optimization suggestions
    pub fn performance_warning(
        operation: &str,
        data_size: usize,
        optimization_advice: &str,
    ) -> Self {
        Self::ComputationError(format!(
            "{operation} may be slow for {data_size} data points - OPTIMIZATION: {optimization_advice}"
        ))
    }
}
