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
    pub const fn new_with_function(file: &'static str, line: u32, function: &'static str) -> Self {
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
    pub const fn new_with_column(file: &'static str, line: u32, column: u32) -> Self {
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
    pub const fn new_full(
        file: &'static str,
        line: u32,
        column: u32,
        function: &'static str,
    ) -> Self {
        Self {
            file,
            line,
            column: Some(column),
            function: Some(function),
        }
    }

    /// Create an error location for the current position (convenience method)
    #[must_use]
    #[inline]
    pub fn here() -> Self {
        Self::new(file!(), line!())
    }
}

impl fmt::Display for ErrorLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.file, self.line)?;
        if let Some(column) = self.column {
            write!(f, ":{column}")?;
        }
        if let Some(function) = self.function {
            write!(f, " in {function}")?;
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
            write!(f, " at {location}")?;
        }
        if let Some(cause) = &self.cause {
            write!(f, "\nCaused by: {cause}")?;
        }
        Ok(())
    }
}

/// Core error type for ``SciRS2``
///
/// This enum is `#[non_exhaustive]`: downstream crates must include a wildcard
/// `_` arm when matching on it, so that adding new variants here is not a
/// breaking change.  The `core_error_non_exhaustive` compile-fail test in
/// `scirs2-stability-tests` guards this contract.
#[derive(Error, Debug, Clone)]
#[non_exhaustive]
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

    /// Allocation error (memory allocation failed)
    #[error("{0}")]
    AllocationError(ErrorContext),

    /// Configuration error (invalid configuration)
    #[error("{0}")]
    ConfigError(ErrorContext),

    /// Invalid argument error
    #[error("{0}")]
    InvalidArgument(ErrorContext),

    /// Invalid input error
    #[error("{0}")]
    InvalidInput(ErrorContext),

    /// Permission error (insufficient permissions)
    #[error("{0}")]
    PermissionError(ErrorContext),

    /// Validation error (input failed validation)
    #[error("{0}")]
    ValidationError(ErrorContext),

    /// Invalid state error (object is in an invalid state)
    #[error("{0}")]
    InvalidState(ErrorContext),

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

    /// Resource error (insufficient or unavailable resources)
    #[error("Resource error: {0}")]
    ResourceError(ErrorContext),

    /// Communication error (network or inter-process communication error)
    #[error("Communication error: {0}")]
    CommunicationError(ErrorContext),

    /// Security error (authentication, authorization, or security-related error)
    #[error("Security error: {0}")]
    SecurityError(ErrorContext),
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

/// Convert from String to CoreError (for parsing errors)
impl From<String> for CoreError {
    fn from(err: String) -> Self {
        CoreError::ValueError(ErrorContext::new(err))
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
            $crate::error::ErrorLocation::new_with_function(file!(), line!(), $function),
        )
    };
}

/// Macro to create a domain error with location information
#[macro_export]
macro_rules! domainerror {
    ($message:expr) => {
        $crate::error::CoreError::DomainError(error_context!($message))
    };
    ($message:expr, $function:expr) => {
        $crate::error::CoreError::DomainError(error_context!($message, $function))
    };
}

/// Macro to create a dimension error with location information
#[macro_export]
macro_rules! dimensionerror {
    ($message:expr) => {
        $crate::error::CoreError::DimensionError(error_context!($message))
    };
    ($message:expr, $function:expr) => {
        $crate::error::CoreError::DimensionError(error_context!($message, $function))
    };
}

/// Macro to create a value error with location information
#[macro_export]
macro_rules! valueerror {
    ($message:expr) => {
        $crate::error::CoreError::ValueError(error_context!($message))
    };
    ($message:expr, $function:expr) => {
        $crate::error::CoreError::ValueError(error_context!($message, $function))
    };
}

/// Macro to create a computation error with location information
#[macro_export]
macro_rules! computationerror {
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
pub fn converterror<E, S>(error: E, message: S) -> CoreError
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
#[allow(dead_code)]
pub fn chainerror<S>(error: CoreError, message: S) -> CoreError
where
    S: Into<String>,
{
    CoreError::ComputationError(
        ErrorContext::new(message)
            .with_location(ErrorLocation::new(file!(), line!()))
            .with_cause(error),
    )
}

/// Error recovery strategies for robust error handling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorRecoveryStrategy {
    /// Fail immediately
    FailFast,
    /// Retry the operation with exponential backoff
    RetryExponential,
    /// Retry the operation with linear backoff
    RetryLinear,
    /// Use a fallback operation
    Fallback,
    /// Skip the failed operation and continue
    Skip,
    /// Use a default value
    UseDefault,
    /// Log the error and continue
    LogAndContinue,
}

/// Error severity levels for prioritized error handling
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    /// Low severity - operation can continue
    Low,
    /// Medium severity - operation can continue with warnings
    Medium,
    /// High severity - operation should be reconsidered
    High,
    /// Critical severity - operation must be stopped
    Critical,
}

/// Error handler configuration for systematic error management
#[derive(Debug, Clone)]
pub struct ErrorHandlerConfig {
    /// Maximum number of retry attempts
    pub max_retries: usize,
    /// Initial retry delay in milliseconds
    pub initial_delay_ms: u64,
    /// Maximum retry delay in milliseconds
    pub max_delay_ms: u64,
    /// Whether to log errors
    pub log_errors: bool,
    /// Error recovery strategy
    pub recovery_strategy: ErrorRecoveryStrategy,
    /// Minimum severity level to handle
    pub min_severity: ErrorSeverity,
}

impl Default for ErrorHandlerConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay_ms: 100,
            max_delay_ms: 5000,
            log_errors: true,
            recovery_strategy: ErrorRecoveryStrategy::FailFast,
            min_severity: ErrorSeverity::Low,
        }
    }
}

/// Enhanced error with severity and recovery information
#[derive(Debug, Clone)]
pub struct EnhancedError {
    /// Core error
    pub error: CoreError,
    /// Error severity
    pub severity: ErrorSeverity,
    /// Suggested recovery strategy
    pub recovery_strategy: ErrorRecoveryStrategy,
    /// Error occurrence timestamp
    pub timestamp: std::time::SystemTime,
    /// Error category for grouping
    pub category: Option<String>,
    /// Error tags for filtering
    pub tags: Vec<String>,
}

impl EnhancedError {
    /// Create a new enhanced error
    pub fn new(error: CoreError, severity: ErrorSeverity) -> Self {
        Self {
            error,
            severity,
            recovery_strategy: ErrorRecoveryStrategy::FailFast,
            timestamp: std::time::SystemTime::now(),
            category: None,
            tags: Vec::new(),
        }
    }

    /// Set recovery strategy
    pub fn with_recovery(mut self, strategy: ErrorRecoveryStrategy) -> Self {
        self.recovery_strategy = strategy;
        self
    }

    /// Set error category
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }

    /// Add error tag
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Check if error has specific tag
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }

    /// Check if error is retryable based on its recovery strategy
    pub fn is_retryable(&self) -> bool {
        matches!(
            self.recovery_strategy,
            ErrorRecoveryStrategy::RetryExponential | ErrorRecoveryStrategy::RetryLinear
        )
    }

    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        !matches!(self.recovery_strategy, ErrorRecoveryStrategy::FailFast)
    }
}

impl fmt::Display for EnhancedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:?}] {}", self.severity, self.error)?;
        if let Some(category) = &self.category {
            write!(f, " (category: {category})")?;
        }
        if !self.tags.is_empty() {
            write!(f, " [tags: {}]", self.tags.join(", "))?;
        }
        Ok(())
    }
}

/// Batch error handling for processing multiple operations
#[derive(Debug, Clone)]
pub struct BatchError {
    /// Individual errors
    pub errors: Vec<EnhancedError>,
    /// Total operations attempted
    pub total_operations: usize,
    /// Number of successful operations
    pub successful_operations: usize,
}

impl BatchError {
    /// Create a new batch error
    pub fn new(total_operations: usize) -> Self {
        Self {
            errors: Vec::new(),
            total_operations,
            successful_operations: 0,
        }
    }

    /// Add an error to the batch
    pub fn add_error(&mut self, error: EnhancedError) {
        self.errors.push(error);
    }

    /// Record a successful operation
    pub fn add_success(&mut self) {
        self.successful_operations += 1;
    }

    /// Get error rate (0.0 to 1.0)
    pub fn error_rate(&self) -> f64 {
        if self.total_operations == 0 {
            0.0
        } else {
            self.errors.len() as f64 / self.total_operations as f64
        }
    }

    /// Get success rate (0.0 to 1.0)
    pub fn success_rate(&self) -> f64 {
        if self.total_operations == 0 {
            0.0
        } else {
            self.successful_operations as f64 / self.total_operations as f64
        }
    }

    /// Check if batch operation was successful (configurable threshold)
    pub fn is_successful(&self, min_success_rate: f64) -> bool {
        self.success_rate() >= min_success_rate
    }

    /// Get errors by severity
    pub fn errors_by_severity(&self, severity: ErrorSeverity) -> Vec<&EnhancedError> {
        self.errors
            .iter()
            .filter(|e| e.severity == severity)
            .collect()
    }

    /// Get errors by category
    pub fn errors_by_category(&self, category: &str) -> Vec<&EnhancedError> {
        self.errors
            .iter()
            .filter(|e| e.category.as_deref() == Some(category))
            .collect()
    }

    /// Get retryable errors
    pub fn retryable_errors(&self) -> Vec<&EnhancedError> {
        self.errors.iter().filter(|e| e.is_retryable()).collect()
    }
}

impl fmt::Display for BatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Batch operation: {}/{} successful ({:.1}% success rate)",
            self.successful_operations,
            self.total_operations,
            self.success_rate() * 100.0
        )?;
        if !self.errors.is_empty() {
            write!(f, "\nErrors ({})", self.errors.len())?;
            for (i, error) in self.errors.iter().enumerate() {
                write!(f, "\n  {}: {}", i + 1, error)?;
            }
        }
        Ok(())
    }
}

/// Result type for enhanced error handling
pub type EnhancedResult<T> = Result<T, EnhancedError>;

/// Result type for batch operations
pub type BatchResult<T> = Result<T, BatchError>;

/// Convenience function to create an enhanced error from a core error
pub fn enhance_error(error: CoreError, severity: ErrorSeverity) -> EnhancedError {
    EnhancedError::new(error, severity)
}

/// Convenience function to create a critical error
pub fn critical_error(error: CoreError) -> EnhancedError {
    EnhancedError::new(error, ErrorSeverity::Critical)
}

/// Convenience function to create a high severity error
pub fn high_error(error: CoreError) -> EnhancedError {
    EnhancedError::new(error, ErrorSeverity::High)
}

/// Convenience function to create a medium severity error
pub fn medium_error(error: CoreError) -> EnhancedError {
    EnhancedError::new(error, ErrorSeverity::Medium)
}

/// Convenience function to create a low severity error
pub fn low_error(error: CoreError) -> EnhancedError {
    EnhancedError::new(error, ErrorSeverity::Low)
}

/// Macro to create an enhanced error with automatic severity detection
#[macro_export]
macro_rules! enhanced_error {
    (critical, $error:expr) => {
        $crate::error::critical_error($error)
    };
    (high, $error:expr) => {
        $crate::error::high_error($error)
    };
    (medium, $error:expr) => {
        $crate::error::medium_error($error)
    };
    (low, $error:expr) => {
        $crate::error::low_error($error)
    };
    ($error:expr) => {
        $crate::error::medium_error($error)
    };
}

/// Macro for batch error handling
#[macro_export]
macro_rules! batch_operation {
    ($total:expr, $operations:block) => {{
        let mut batch_error = $crate::error::BatchError::new($total);
        let result = $operations;
        if batch_error.errors.is_empty() {
            Ok(result)
        } else {
            Err(batch_error)
        }
    }};
}

/// Macro for error recovery with retry logic
#[macro_export]
macro_rules! with_retry {
    ($operation:expr, $max_retries:expr) => {{
        let mut attempts = 0;
        let mut last_error = None;

        loop {
            match $operation {
                Ok(result) => break Ok(result),
                Err(error) => {
                    attempts += 1;
                    last_error = Some(error);

                    if attempts >= $max_retries {
                        break Err(last_error.expect("Operation failed"));
                    }

                    // Simple linear backoff
                    std::thread::sleep(std::time::Duration::from_millis(attempts * 100));
                }
            }
        }
    }};
}
