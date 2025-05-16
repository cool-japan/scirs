//! Error types for the SciRS2 integration module

use thiserror::Error;

/// Integration error type
#[derive(Error, Debug, Clone)]
pub enum IntegrateError {
    /// Computation error (generic error)
    #[error("Computation error: {0}")]
    ComputationError(String),

    /// Convergence error (algorithm did not converge)
    #[error("Convergence error: {0}")]
    ConvergenceError(String),

    /// Value error (invalid value)
    #[error("Value error: {0}")]
    ValueError(String),

    /// Not implemented error
    #[error("Not implemented: {0}")]
    NotImplementedError(String),

    /// Linear system solver error
    #[error("Linear solve error: {0}")]
    LinearSolveError(String),

    /// Dimension mismatch error
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// Method switching error
    #[error("Method switching error: {0}")]
    MethodSwitchingError(String),

    /// Step size too small error
    #[error("Step size too small: {0}")]
    StepSizeTooSmall(String),
}

/// Result type for integration operations
pub type IntegrateResult<T> = Result<T, IntegrateError>;
