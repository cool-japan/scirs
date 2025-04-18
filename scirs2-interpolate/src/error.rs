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

    /// Not implemented error
    #[error("Not implemented: {0}")]
    NotImplementedError(String),
}

/// Result type for interpolation operations
pub type InterpolateResult<T> = Result<T, InterpolateError>;
