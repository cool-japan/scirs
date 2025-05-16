//! Error types for the SciRS2 special functions module

use thiserror::Error;

/// Special functions error type
#[derive(Error, Debug)]
pub enum SpecialError {
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

    /// Convergence error (algorithm did not converge)
    #[error("Convergence error: {0}")]
    ConvergenceError(String),
}

/// Result type for special functions operations
pub type SpecialResult<T> = Result<T, SpecialError>;
