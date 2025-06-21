//! Error types for the ``SciRS2`` core module
//!
//! This module provides common error types used throughout the ``SciRS2`` ecosystem.

use std::fmt;
use thiserror::Error;

/// Location information for error context
#[derive(Debug, Clone)]
pub struct ErrorLocation {
    /// File where the error occurred
    pub file: &'static str,
    /// Line number where the error occurred
    pub line: u32,
    /// Column number where the error occurred
    pub column: Option<u32>,
    /// Function where the error occurred
    pub function: Option<&'static str>,
}

impl ErrorLocation {
    /// Create a new error location
    #[must_use]
    #[inline]
    pub const fn new(file: &'static str, line: u32) -> Self {
        Self {
            file,
            line,
            column: None,
            function: None,
        }
    }

    /// Create a new error location with function information
    #[must_use]
    #[inline]
    pub const fn with_function(file: &'static str, line: u32, function: &'static str) -> Self {
        Self {
            file,
            line,
            column: None,
            function: Some(function),
        }
    }

    /// Create a new error location with column information
    #[must_use]
    #[inline]
    pub const fn with_column(file: &'static str, line: u32, column: u32) -> Self {
        Self {
            file,
            line,
            column: Some(column),
            function: None,
        }
    }

    /// Create a new error location with function and column information
    #[must_use]
    #[inline]
    pub const fn full(file: &'static str, line: u32, column: u32, function: &'static str) -> Self {
        Self {
            file,
            line,
            column: Some(column),
            function: Some(function),
        }
    }
}

impl fmt::Display for ErrorLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.file, self.line)?;
        if let Some(column) = self.column {
            write!(f, ":{}", column)?;
        }
        if let Some(function) = self.function {
            write!(f, " in {}", function)?;
        }
        Ok(())
    }
}

/// Error context containing additional information about an error
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Error message
    pub message: String,
    /// Location where the error occurred
    pub location: Option<ErrorLocation>,
    /// Cause of the error
    pub cause: Option<Box<CoreError>>,
}

impl ErrorContext {
    /// Create a new error context
    #[must_use]
    pub fn new<S: Into<String>>(message: S) -> Self {
        Self {
            message: message.into(),
            location: None,
            cause: None,
        }
    }

    /// Add location information to the error context
    #[must_use]
    pub fn with_location(mut self, location: ErrorLocation) -> Self {
        self.location = Some(location);
        self
    }

    /// Add a cause to the error context
    #[must_use]
    pub fn with_cause(mut self, cause: CoreError) -> Self {
        self.cause = Some(Box::new(cause));
        self
    }
}

impl fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)?;
        if let Some(location) = &self.location {
            write!(f, " at {}", location)?;
        }
        if let Some(cause) = &self.cause {
            write!(f, "\nCaused by: {}", cause)?;
        }
        Ok(())
    }
}

/// Core error type for ``SciRS2``
#[derive(Error, Debug, Clone)]
pub enum CoreError {
    /// Computation error (generic error)
    #[error("{0}")]
    ComputationError(ErrorContext),

    /// Domain error (input outside valid domain)
    #[error("{0}")]
    DomainError(ErrorContext),

    /// Dispatch error (array protocol dispatch failed)
    #[error("{0}")]
    DispatchError(ErrorContext),

    /// Convergence error (algorithm did not converge)
    #[error("{0}")]
    ConvergenceError(ErrorContext),

    /// Dimension mismatch error
    #[error("{0}")]
    DimensionError(ErrorContext),

    /// Shape error (matrices/arrays have incompatible shapes)
    #[error("{0}")]
    ShapeError(ErrorContext),

    /// Out of bounds error
    #[error("{0}")]
    IndexError(ErrorContext),

    /// Value error (invalid value)
    #[error("{0}")]
    ValueError(ErrorContext),

    /// Type error (invalid type)
    #[error("{0}")]
    TypeError(ErrorContext),

    /// Not implemented error
    #[error("{0}")]
    NotImplementedError(ErrorContext),

    /// Implementation error (method exists but not fully implemented yet)
    #[error("{0}")]
    ImplementationError(ErrorContext),

    /// Memory error (could not allocate memory)
    #[error("{0}")]
    MemoryError(ErrorContext),

    /// Configuration error (invalid configuration)
    #[error("{0}")]
    ConfigError(ErrorContext),

    /// Invalid argument error
    #[error("{0}")]
    InvalidArgument(ErrorContext),

    /// Permission error (insufficient permissions)
    #[error("{0}")]
    PermissionError(ErrorContext),

    /// Validation error (input failed validation)
    #[error("{0}")]
    ValidationError(ErrorContext),

    /// JIT compilation error (error during JIT compilation)
    #[error("{0}")]
    JITError(ErrorContext),

    /// JSON error
    #[error("JSON error: {0}")]
    JSONError(ErrorContext),

    /// IO error
    #[error("IO error: {0}")]
    IoError(ErrorContext),

    /// Scheduler error (error in work-stealing scheduler)
    #[error("Scheduler error: {0}")]
    SchedulerError(ErrorContext),

    /// Timeout error (operation timed out)
    #[error("Timeout error: {0}")]
    TimeoutError(ErrorContext),

    /// Compression error (error during compression/decompression)
    #[error("Compression error: {0}")]
    CompressionError(ErrorContext),

    /// Invalid shape error (array shape is invalid)
    #[error("Invalid shape: {0}")]
    InvalidShape(ErrorContext),

    /// Device error (GPU/hardware device error)
    #[error("Device error: {0}")]
    DeviceError(ErrorContext),

    /// Mutex error (mutex poisoning or lock error)
    #[error("Mutex error: {0}")]
    MutexError(ErrorContext),

    /// Thread error (threading error)
    #[error("Thread error: {0}")]
    ThreadError(ErrorContext),

    /// Stream error (streaming operation error)
    #[error("Stream error: {0}")]
    StreamError(ErrorContext),

    /// End of stream error (stream ended unexpectedly)
    #[error("End of stream: {0}")]
    EndOfStream(ErrorContext),
}

/// Result type alias for core operations
pub type CoreResult<T> = Result<T, CoreError>;

/// Convert from std::io::Error to CoreError
impl From<std::io::Error> for CoreError {
    fn from(err: std::io::Error) -> Self {
        CoreError::IoError(ErrorContext::new(format!("IO error: {err}")))
    }
}

/// Convert from serde_json::Error to CoreError
#[cfg(feature = "serialization")]
impl From<serde_json::Error> for CoreError {
    fn from(err: serde_json::Error) -> Self {
        CoreError::JSONError(ErrorContext::new(format!("JSON error: {err}")))
    }
}

/// Convert from OperationError to CoreError
impl From<crate::array_protocol::OperationError> for CoreError {
    fn from(err: crate::array_protocol::OperationError) -> Self {
        use crate::array_protocol::OperationError;
        match err {
            // Preserving NotImplemented for compatibility with older code,
            // but it will eventually be replaced with NotImplementedError
            OperationError::NotImplemented(msg) => {
                CoreError::NotImplementedError(ErrorContext::new(msg))
            }
            OperationError::ShapeMismatch(msg) => CoreError::ShapeError(ErrorContext::new(msg)),
            OperationError::TypeMismatch(msg) => CoreError::TypeError(ErrorContext::new(msg)),
            OperationError::Other(msg) => CoreError::ComputationError(ErrorContext::new(msg)),
        }
    }
}

/// Macro to create a new error context with location information
///
/// # Example
///
/// ```rust
/// use scirs2_core::error_context;
/// use scirs2_core::error::{CoreResult, CoreError};
///
/// fn example() -> CoreResult<()> {
///     let condition = false;
///     if condition {
///         return Err(CoreError::ComputationError(error_context!("An error occurred")));
///     }
///     Ok(())
/// }
/// ```
#[macro_export]
macro_rules! error_context {
    ($message:expr) => {
        $crate::error::ErrorContext::new($message)
            .with_location($crate::error::ErrorLocation::new(file!(), line!()))
    };
    ($message:expr, $function:expr) => {
        $crate::error::ErrorContext::new($message).with_location(
            $crate::error::ErrorLocation::with_function(file!(), line!(), $function),
        )
    };
}

/// Macro to create a domain error with location information
#[macro_export]
macro_rules! domain_error {
    ($message:expr) => {
        $crate::error::CoreError::DomainError(error_context!($message))
    };
    ($message:expr, $function:expr) => {
        $crate::error::CoreError::DomainError(error_context!($message, $function))
    };
}

/// Macro to create a dimension error with location information
#[macro_export]
macro_rules! dimension_error {
    ($message:expr) => {
        $crate::error::CoreError::DimensionError(error_context!($message))
    };
    ($message:expr, $function:expr) => {
        $crate::error::CoreError::DimensionError(error_context!($message, $function))
    };
}

/// Macro to create a value error with location information
#[macro_export]
macro_rules! value_error {
    ($message:expr) => {
        $crate::error::CoreError::ValueError(error_context!($message))
    };
    ($message:expr, $function:expr) => {
        $crate::error::CoreError::ValueError(error_context!($message, $function))
    };
}

/// Macro to create a computation error with location information
#[macro_export]
macro_rules! computation_error {
    ($message:expr) => {
        $crate::error::CoreError::ComputationError(error_context!($message))
    };
    ($message:expr, $function:expr) => {
        $crate::error::CoreError::ComputationError(error_context!($message, $function))
    };
}

/// Checks if a condition is true, otherwise returns a domain error
///
/// # Arguments
///
/// * `condition` - The condition to check
/// * `message` - The error message if the condition is false
///
/// # Returns
///
/// * `Ok(())` if the condition is true
/// * `Err(CoreError::DomainError)` if the condition is false
///
/// # Errors
///
/// Returns `CoreError::DomainError` if the condition is false.
pub fn check_domain<S: Into<String>>(condition: bool, message: S) -> CoreResult<()> {
    if condition {
        Ok(())
    } else {
        Err(CoreError::DomainError(
            ErrorContext::new(message).with_location(ErrorLocation::new(file!(), line!())),
        ))
    }
}

/// Checks dimensions
///
/// # Arguments
///
/// * `condition` - The condition to check
/// * `message` - The error message if the condition is false
///
/// # Returns
///
/// * `Ok(())` if the condition is true
/// * `Err(CoreError::DimensionError)` if the condition is false
///
/// # Errors
///
/// Returns `CoreError::DimensionError` if the condition is false.
pub fn check_dimensions<S: Into<String>>(condition: bool, message: S) -> CoreResult<()> {
    if condition {
        Ok(())
    } else {
        Err(CoreError::DimensionError(
            ErrorContext::new(message).with_location(ErrorLocation::new(file!(), line!())),
        ))
    }
}

/// Checks if a value is valid
///
/// # Arguments
///
/// * `condition` - The condition to check
/// * `message` - The error message if the condition is false
///
/// # Returns
///
/// * `Ok(())` if the condition is true
/// * `Err(CoreError::ValueError)` if the condition is false
///
/// # Errors
///
/// Returns `CoreError::ValueError` if the condition is false.
pub fn check_value<S: Into<String>>(condition: bool, message: S) -> CoreResult<()> {
    if condition {
        Ok(())
    } else {
        Err(CoreError::ValueError(
            ErrorContext::new(message).with_location(ErrorLocation::new(file!(), line!())),
        ))
    }
}

/// Checks if a value is valid according to a validator function
///
/// # Arguments
///
/// * `value` - The value to validate
/// * `validator` - A function that returns true if the value is valid
/// * `message` - The error message if the value is invalid
///
/// # Returns
///
/// * `Ok(value)` if the value is valid
/// * `Err(CoreError::ValidationError)` if the value is invalid
///
/// # Errors
///
/// Returns `CoreError::ValidationError` if the validator function returns false.
pub fn validate<T, F, S>(value: T, validator: F, message: S) -> CoreResult<T>
where
    F: FnOnce(&T) -> bool,
    S: Into<String>,
{
    if validator(&value) {
        Ok(value)
    } else {
        Err(CoreError::ValidationError(
            ErrorContext::new(message).with_location(ErrorLocation::new(file!(), line!())),
        ))
    }
}

/// Convert an error from one type to a CoreError
///
/// # Arguments
///
/// * `error` - The error to convert
/// * `message` - A message describing the context of the error
///
/// # Returns
///
/// * A CoreError with the original error as its cause
#[must_use]
pub fn convert_error<E, S>(error: E, message: S) -> CoreError
where
    E: std::error::Error + 'static,
    S: Into<String>,
{
    // Create a computation error that contains the original error
    // We combine the provided message with the error's own message for extra context
    let message_str = message.into();
    let error_message = format!("{message_str} | Original error: {error}");

    // For I/O errors we have direct conversion via From trait implementation
    // but we can't use it directly due to the generic bounds.
    // In a real implementation, you would use a match or if statement with
    // type_id or another approach to distinguish error types.

    // For simplicity, we'll just use ComputationError as a general case
    CoreError::ComputationError(
        ErrorContext::new(error_message).with_location(ErrorLocation::new(file!(), line!())),
    )
}

/// Create an error chain by adding a new error context
///
/// # Arguments
///
/// * `error` - The error to chain
/// * `message` - A message describing the context of the error
///
/// # Returns
///
/// * A CoreError with the original error as its cause
#[must_use]
pub fn chain_error<S>(error: CoreError, message: S) -> CoreError
where
    S: Into<String>,
{
    CoreError::ComputationError(
        ErrorContext::new(message)
            .with_location(ErrorLocation::new(file!(), line!()))
            .with_cause(error),
    )
}
