//! Error types for the SciRS2 interpolation module

use thiserror::Error;

/// Interpolation error type
#[derive(Error, Debug)]
pub enum InterpolateError {
    /// Computation error (generic error)
    #[error("Computation error: {0}")]
    ComputationError(String),

    /// Domain error (input outside valid domain)
    #[error("Domain error: {0}")]
    DomainError(String),

    /// Value error (invalid value)
    #[error("Value error: {0}")]
    ValueError(String),

    /// Shape error (ndarray shape mismatch)
    #[error("Shape error: {0}")]
    ShapeError(String),

    /// Not implemented error
    #[error("Not implemented: {0}")]
    NotImplementedError(String),

    /// Feature not implemented (alias for NotImplementedError)
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
}

impl From<ndarray::ShapeError> for InterpolateError {
    fn from(err: ndarray::ShapeError) -> Self {
        InterpolateError::ShapeError(err.to_string())
    }
}

/// Result type for interpolation operations
pub type InterpolateResult<T> = Result<T, InterpolateError>;
