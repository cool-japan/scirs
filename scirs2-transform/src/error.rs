//! Error types for the data transformation module

use thiserror::Error;

/// Error type for data transformation operations
#[derive(Error, Debug)]
pub enum TransformError {
    /// Invalid input data
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Transformation error
    #[error("Transformation error: {0}")]
    TransformationError(String),

    /// Core error
    #[error("Core error: {0}")]
    CoreError(#[from] scirs2_core::error::CoreError),

    /// Linear algebra error
    #[error("Linear algebra error: {0}")]
    LinalgError(#[from] scirs2_linalg::error::LinalgError),

    /// FFT error
    #[error("FFT error: {0}")]
    FFTError(#[from] scirs2_fft::error::FFTError),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Computation error
    #[error("Computation error: {0}")]
    ComputationError(String),

    /// Model not fitted error
    #[error("Model not fitted: {0}")]
    NotFitted(String),

    /// Feature not enabled error
    #[error("Feature not enabled: {0}")]
    FeatureNotEnabled(String),

    /// GPU error
    #[error("GPU error: {0}")]
    GpuError(String),

    /// Distributed processing error
    #[error("Distributed processing error: {0}")]
    DistributedError(String),

    /// Monitoring error
    #[error("Monitoring error: {0}")]
    MonitoringError(String),

    /// Memory allocation error
    #[error("Memory error: {0}")]
    MemoryError(String),

    /// Convergence failure in iterative algorithms
    #[error("Convergence error: {0}")]
    ConvergenceError(String),

    /// Data quality or validation error
    #[error("Data validation error: {0}")]
    DataValidationError(String),

    /// Threading or parallel processing error
    #[error("Parallel processing error: {0}")]
    ParallelError(String),

    /// Configuration validation error
    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    /// Timeout error for long-running operations
    #[error("Timeout error: {0}")]
    TimeoutError(String),

    /// SIMD operation error
    #[error("SIMD error: {0}")]
    SimdError(String),

    /// Streaming data pipeline error
    #[error("Streaming error: {0}")]
    StreamingError(String),

    /// Cross-validation error
    #[error("Cross-validation error: {0}")]
    CrossValidationError(String),

    /// Prometheus error
    #[cfg(feature = "monitoring")]
    #[error("Prometheus error: {0}")]
    PrometheusError(#[from] prometheus::Error),

    /// Serialization error
    #[cfg(feature = "distributed")]
    #[error("Serialization error: {0}")]
    SerializationError(#[from] bincode::Error),

    /// Not implemented error
    #[error("Not implemented: {0}")]
    NotImplemented(String),

    /// Parse error
    #[error("Parse error: {0}")]
    ParseError(String),

    /// PyTorch/tch error
    #[cfg(feature = "auto-feature-engineering")]
    #[error("PyTorch error: {0}")]
    TchError(#[from] tch::TchError),

    /// Other error
    #[error("Error: {0}")]
    Other(String),
}

/// Result type for data transformation operations
pub type Result<T> = std::result::Result<T, TransformError>;

/// Context trait for adding context to errors
pub trait ErrorContext<T> {
    /// Add context to an error
    fn context(self, msg: &str) -> Result<T>;

    /// Add context with a format string
    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String;
}

impl<T> ErrorContext<T> for Result<T> {
    fn context(self, msg: &str) -> Result<T> {
        self.map_err(|e| TransformError::Other(format!("{msg}: {e}")))
    }

    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|e| TransformError::Other(format!("{}: {e}", f())))
    }
}

/// Error kind for categorizing errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorKind {
    /// Input validation errors
    Validation,
    /// Computation errors
    Computation,
    /// Configuration errors
    Configuration,
    /// Resource errors (memory, GPU, etc.)
    Resource,
    /// External errors (IO, network, etc.)
    External,
    /// Internal errors (bugs, not implemented, etc.)
    Internal,
}

impl TransformError {
    /// Get the error kind
    pub fn kind(&self) -> ErrorKind {
        match self {
            TransformError::InvalidInput(_)
            | TransformError::DataValidationError(_)
            | TransformError::ConfigurationError(_) => ErrorKind::Validation,

            TransformError::ComputationError(_)
            | TransformError::TransformationError(_)
            | TransformError::ConvergenceError(_)
            | TransformError::SimdError(_) => ErrorKind::Computation,

            TransformError::MemoryError(_)
            | TransformError::GpuError(_)
            | TransformError::TimeoutError(_) => ErrorKind::Resource,

            TransformError::IoError(_)
            | TransformError::DistributedError(_)
            | TransformError::StreamingError(_)
            | TransformError::MonitoringError(_) => ErrorKind::External,

            TransformError::NotImplemented(_)
            | TransformError::NotFitted(_)
            | TransformError::FeatureNotEnabled(_)
            | TransformError::Other(_) => ErrorKind::Internal,

            TransformError::CoreError(_)
            | TransformError::LinalgError(_)
            | TransformError::FFTError(_) => ErrorKind::External,

            TransformError::ParallelError(_)
            | TransformError::CrossValidationError(_)
            | TransformError::ParseError(_) => ErrorKind::Computation,

            #[cfg(feature = "monitoring")]
            TransformError::PrometheusError(_) => ErrorKind::External,

            #[cfg(feature = "distributed")]
            TransformError::SerializationError(_) => ErrorKind::External,

            #[cfg(feature = "auto-feature-engineering")]
            TransformError::TchError(_) => ErrorKind::Computation,
        }
    }

    /// Check if the error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self.kind() {
            ErrorKind::Validation | ErrorKind::Configuration => false,
            ErrorKind::Resource | ErrorKind::External => true,
            ErrorKind::Computation => true, // May be recoverable with different params
            ErrorKind::Internal => false,
        }
    }

    /// Check if the error should trigger a retry
    pub fn should_retry(&self) -> bool {
        matches!(
            self,
            TransformError::TimeoutError(_)
                | TransformError::MemoryError(_)
                | TransformError::DistributedError(_)
                | TransformError::StreamingError(_)
                | TransformError::ParallelError(_)
                | TransformError::IoError(_)
        )
    }

    /// Get user-friendly error message
    pub fn user_message(&self) -> String {
        match self {
            TransformError::InvalidInput(_) => "Invalid input data provided".to_string(),
            TransformError::NotFitted(_) => "Model must be fitted before use".to_string(),
            TransformError::MemoryError(_) => "Insufficient memory for operation".to_string(),
            TransformError::TimeoutError(_) => "Operation timed out".to_string(),
            TransformError::FeatureNotEnabled(_) => "Required feature is not enabled".to_string(),
            _ => "An error occurred during transformation".to_string(),
        }
    }
}
