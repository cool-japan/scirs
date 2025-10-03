//! Error types for the SciRS2 statistics module

use scirs2_core::error::{CoreError, ErrorContext, ErrorLocation};
use thiserror::Error;

/// Statistics error type
#[derive(Error, Debug, Clone)]
pub enum StatsError {
    /// Computation error (generic error)
    #[error("Computation error: {0}")]
    ComputationError(String),

    /// Domain error (input outside valid domain)
    #[error("Domain error: {0}")]
    DomainError(String),

    /// Dimension mismatch error
    #[error("Dimension mismatch error: {0}")]
    DimensionMismatch(String),

    /// Value error (invalid value)
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    /// Not implemented error
    #[error("Not implemented: {0}")]
    NotImplementedError(String),

    /// Convergence error (algorithm failed to converge)
    #[error("Convergence error: {0}")]
    ConvergenceError(String),

    /// Insufficient data error
    #[error("Insufficient data: {0}")]
    InsufficientData(String),

    /// Invalid input error
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Not implemented (alias for backwards compatibility)
    #[error("Not implemented: {0}")]
    NotImplemented(String),

    /// Core error (propagated from scirs2-core)
    #[error("{0}")]
    CoreError(#[from] CoreError),

    /// Random distribution error
    #[error("Random distribution error: {0}")]
    DistributionError(String),
}

// The #[from] attribute in the CoreError variant handles the conversion automatically

// NOTE: rand, distr: uniform::Error API has changed, commenting out for now
// impl From<scirs2_core::random::uniform::Error> for StatsError {
//     fn from(err: rand, distr: uniform::Error) -> Self {
//         StatsError::DistributionError(format!("Uniform distribution error: {}", err))
//     }
// }

/// Helper trait for adding context and recovery suggestions to errors
pub trait StatsErrorExt {
    /// Add context information to the error
    fn context<S: Into<String>>(self, context: S) -> Self;

    /// Add a recovery suggestion to the error
    fn suggestion<S: Into<String>>(self, suggestion: S) -> Self;
}

impl StatsError {
    /// Create a computation error with context
    pub fn computation<S: Into<String>>(message: S) -> Self {
        StatsError::ComputationError(message.into())
    }

    /// Create a domain error with context
    pub fn domain<S: Into<String>>(message: S) -> Self {
        StatsError::DomainError(message.into())
    }

    /// Create a dimension mismatch error with context
    pub fn dimension_mismatch<S: Into<String>>(message: S) -> Self {
        StatsError::DimensionMismatch(message.into())
    }

    /// Create an invalid argument error with context
    pub fn invalid_argument<S: Into<String>>(message: S) -> Self {
        StatsError::InvalidArgument(message.into())
    }

    /// Create a not implemented error with context
    pub fn not_implemented<S: Into<String>>(message: S) -> Self {
        StatsError::NotImplementedError(message.into())
    }

    /// Create an insufficient data error with context
    pub fn insufficientdata<S: Into<String>>(message: S) -> Self {
        StatsError::InsufficientData(message.into())
    }

    /// Create an invalid input error with context
    pub fn invalid_input<S: Into<String>>(message: S) -> Self {
        StatsError::InvalidInput(message.into())
    }

    /// Add recovery suggestions based on error type
    pub fn with_suggestion(&self) -> String {
        match self {
            StatsError::DomainError(msg) => {
                if msg.contains("must be positive") {
                    format!(
                        "{msg}
Suggestion: Ensure the value is greater than 0"
                    )
                } else if msg.contains("probability") {
                    format!(
                        "{msg}
Suggestion: Probability values must be between 0 and 1 (inclusive)"
                    )
                } else if msg.contains("degrees of freedom") {
                    format!(
                        "{msg}
Suggestion: Degrees of freedom must be a positive value"
                    )
                } else {
                    msg.clone()
                }
            }
            StatsError::DimensionMismatch(msg) => {
                if msg.contains("same length") {
                    format!(
                        "{msg}
Suggestion: Ensure both arrays have the same number of elements"
                    )
                } else if msg.contains("square matrix") {
                    format!(
                        "{msg}
Suggestion: The input matrix must have equal number of rows and columns"
                    )
                } else {
                    format!(
                        "{msg}
Suggestion: Check that input dimensions match the function requirements"
                    )
                }
            }
            StatsError::InvalidArgument(msg) => {
                if msg.contains("empty") {
                    format!(
                        "{msg}
Suggestion: Provide a non-empty array or collection"
                    )
                } else if msg.contains("NaN") || msg.contains("nan") {
                    format!(
                        "{msg}
Suggestion: Remove or handle NaN values before computation"
                    )
                } else if msg.contains("infinite") || msg.contains("inf") {
                    format!(
                        "{msg}
Suggestion: Check for and handle infinite values in your data"
                    )
                } else {
                    format!(
                        "{msg}
Suggestion: Verify that all input arguments meet the function requirements"
                    )
                }
            }
            StatsError::NotImplementedError(msg) => {
                format!("{msg}
Suggestion: This feature is not yet available. Consider using an alternative method or check for updates")
            }
            StatsError::ComputationError(msg) => {
                if msg.contains("overflow") {
                    format!(
                        "{msg}
Suggestion: Try scaling your input data or using a more numerically stable algorithm"
                    )
                } else if msg.contains("convergence") {
                    format!(
                        "{msg}
Suggestion: Try adjusting convergence parameters or using different initial values"
                    )
                } else {
                    format!(
                        "{msg}
Suggestion: Check input data for numerical issues or extreme values"
                    )
                }
            }
            StatsError::ConvergenceError(msg) => {
                format!("{msg}
Suggestion: Try adjusting convergence parameters, using different initial values, or increasing the maximum number of iterations")
            }
            StatsError::InsufficientData(msg) => {
                format!(
                    "{msg}
Suggestion: Increase sample size or use methods designed for small datasets"
                )
            }
            StatsError::InvalidInput(msg) => {
                format!(
                    "{msg}
Suggestion: Check input format and ensure data meets function requirements"
                )
            }
            StatsError::NotImplemented(msg) => {
                format!("{msg}
Suggestion: This feature is not yet available. Consider using an alternative method or check for updates")
            }
            StatsError::CoreError(err) => {
                format!(
                    "{err}
Suggestion: {}",
                    "Refer to the core error for more details"
                )
            }
            StatsError::DistributionError(msg) => {
                format!(
                    "{msg}
Suggestion: Check distribution parameters and ensure they are within valid ranges"
                )
            }
        }
    }
}

/// Result type for statistics operations
pub type StatsResult<T> = Result<T, StatsError>;

/// Create a function to convert from StatsResult to CoreError::ValidationError
#[allow(dead_code)]
pub fn convert_to_validation_error<T, S: Into<String>>(
    result: StatsResult<T>,
    message: S,
) -> Result<T, CoreError> {
    match result {
        Ok(val) => Ok(val),
        Err(err) => Err(CoreError::ValidationError(
            ErrorContext::new(format!("{}: {err}", message.into()))
                .with_location(ErrorLocation::new(file!(), line!())),
        )),
    }
}
