//! Error types for the ML optimization module

use std::error::Error;
use std::fmt;

/// Error type for ML optimization operations
#[derive(Debug)]
pub enum OptimError {
    /// Invalid configuration
    InvalidConfig(String),
    /// Invalid parameter
    InvalidParameter(String),
    /// Optimization error
    OptimizationError(String),
    /// Dimension mismatch error
    DimensionMismatch(String),
    /// Privacy budget exhausted
    PrivacyBudgetExhausted {
        consumed_epsilon: f64,
        target_epsilon: f64,
    },
    /// Invalid privacy configuration
    InvalidPrivacyConfig(String),
    /// Privacy accounting error
    PrivacyAccountingError(String),
    /// Configuration error
    ConfigurationError(String),
    /// Invalid state error
    InvalidState(String),
    /// Monitoring error
    MonitoringError(String),
    /// Analysis error
    AnalysisError(String),
    /// Analytics error
    AnalyticsError(String),
    /// Unsupported operation
    UnsupportedOperation(String),
    /// Resource error
    ResourceError(String),
    /// Resource unavailable error
    ResourceUnavailable(String),
    /// Execution error
    ExecutionError(String),
    /// Environment error
    Environment(String),
    /// Lock error
    LockError(String),
    /// Thread error
    ThreadError(String),
    /// Computation error
    ComputationError(String),
    /// Plugin still in use error
    PluginStillInUse(String),
    /// Missing dependency error
    MissingDependency(String),
    /// Plugin not found error
    PluginNotFound(String),
    /// Plugin disabled error
    PluginDisabled(String),
    /// Plugin load error
    PluginLoadError(String),
    /// Plugin in maintenance error
    PluginInMaintenance(String),
    /// Unsupported data type error
    UnsupportedDataType(String),
    /// Input/Output error
    IO(std::io::Error),
    /// Other error
    Other(String),
}

/// Alias for backward compatibility
pub type OptimizerError = OptimError;

impl fmt::Display for OptimError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            OptimError::InvalidConfig(msg) => write!(f, "Invalid configuration: {msg}"),
            OptimError::InvalidParameter(msg) => write!(f, "Invalid parameter: {msg}"),
            OptimError::OptimizationError(msg) => write!(f, "Optimization error: {msg}"),
            OptimError::DimensionMismatch(msg) => write!(f, "Dimension mismatch: {msg}"),
            OptimError::PrivacyBudgetExhausted {
                consumed_epsilon,
                target_epsilon,
            } => {
                write!(
                    f,
                    "Privacy budget exhausted: consumed ε={consumed_epsilon:.4}, target ε={target_epsilon:.4}"
                )
            }
            OptimError::InvalidPrivacyConfig(msg) => {
                write!(f, "Invalid privacy configuration: {msg}")
            }
            OptimError::PrivacyAccountingError(msg) => {
                write!(f, "Privacy accounting error: {msg}")
            }
            OptimError::ConfigurationError(msg) => {
                write!(f, "Configuration error: {msg}")
            }
            OptimError::InvalidState(msg) => {
                write!(f, "Invalid state error: {msg}")
            }
            OptimError::MonitoringError(msg) => {
                write!(f, "Monitoring error: {msg}")
            }
            OptimError::AnalysisError(msg) => {
                write!(f, "Analysis error: {msg}")
            }
            OptimError::AnalyticsError(msg) => {
                write!(f, "Analytics error: {msg}")
            }
            OptimError::UnsupportedOperation(msg) => {
                write!(f, "Unsupported operation: {msg}")
            }
            OptimError::ResourceError(msg) => {
                write!(f, "Resource error: {msg}")
            }
            OptimError::ResourceUnavailable(msg) => {
                write!(f, "Resource unavailable: {msg}")
            }
            OptimError::ExecutionError(msg) => {
                write!(f, "Execution error: {msg}")
            }
            OptimError::Environment(msg) => {
                write!(f, "Environment error: {msg}")
            }
            OptimError::LockError(msg) => {
                write!(f, "Lock error: {msg}")
            }
            OptimError::ThreadError(msg) => {
                write!(f, "Thread error: {msg}")
            }
            OptimError::ComputationError(msg) => {
                write!(f, "Computation error: {msg}")
            }
            OptimError::PluginStillInUse(msg) => {
                write!(f, "Plugin still in use: {msg}")
            }
            OptimError::MissingDependency(msg) => {
                write!(f, "Missing dependency: {msg}")
            }
            OptimError::PluginNotFound(msg) => {
                write!(f, "Plugin not found: {msg}")
            }
            OptimError::PluginDisabled(msg) => {
                write!(f, "Plugin disabled: {msg}")
            }
            OptimError::PluginLoadError(msg) => {
                write!(f, "Plugin load error: {msg}")
            }
            OptimError::PluginInMaintenance(msg) => {
                write!(f, "Plugin in maintenance: {msg}")
            }
            OptimError::UnsupportedDataType(msg) => {
                write!(f, "Unsupported data type: {msg}")
            }
            OptimError::IO(err) => {
                write!(f, "I/O error: {err}")
            }
            OptimError::Other(msg) => write!(f, "Error: {msg}"),
        }
    }
}

impl Error for OptimError {}

/// From implementations for common error types
impl From<std::time::SystemTimeError> for OptimError {
    fn from(error: std::time::SystemTimeError) -> Self {
        OptimError::Other(format!("System time error: {error}"))
    }
}

impl From<ndarray::ShapeError> for OptimError {
    fn from(error: ndarray::ShapeError) -> Self {
        OptimError::DimensionMismatch(format!("Shape error: {error}"))
    }
}

impl From<serde_json::Error> for OptimError {
    fn from(error: serde_json::Error) -> Self {
        OptimError::Other(format!("Serde JSON error: {error}"))
    }
}

impl From<std::io::Error> for OptimError {
    fn from(error: std::io::Error) -> Self {
        OptimError::IO(error)
    }
}

/// Result type for ML optimization operations
pub type Result<T> = std::result::Result<T, OptimError>;
