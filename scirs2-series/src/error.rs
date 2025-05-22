//! Error types for the time series module

// No imports needed here, thiserror handles the implementations
use scirs2_core::error::CoreError;
use thiserror::Error;

/// Error type for time series operations
#[derive(Debug, Error)]
pub enum TimeSeriesError {
    /// Invalid input data
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Insufficient data for operation
    #[error("Insufficient data: {message}. Need at least {required} observations, got {actual}")]
    InsufficientData {
        /// Error message
        message: String,
        /// Required number of observations
        required: usize,
        /// Actual number of observations
        actual: usize,
    },

    /// Invalid model configuration
    #[error("Invalid model configuration: {0}")]
    InvalidModel(String),

    /// Model fitting error
    #[error("Model fitting error: {0}")]
    FittingError(String),

    /// Forecasting error
    #[error("Forecasting error: {0}")]
    ForecastingError(String),

    /// Decomposition error
    #[error("Decomposition error: {0}")]
    DecompositionError(String),

    /// Feature extraction error
    #[error("Feature extraction error: {0}")]
    FeatureExtractionError(String),

    /// Statistical error
    #[error("Statistical error: {0}")]
    StatisticalError(String),

    /// Optimization error
    #[error("Optimization error: {0}")]
    OptimizationError(String),

    /// Convergence error
    #[error("Failed to converge after {iterations} iterations")]
    ConvergenceError {
        /// Number of iterations attempted
        iterations: usize,
    },

    /// Numerical instability
    #[error("Numerical instability: {0}")]
    NumericalInstability(String),

    /// Computation error
    #[error("Computation error: {0}")]
    ComputationError(String),

    /// Dimension mismatch
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension
        expected: usize,
        /// Actual dimension
        actual: usize,
    },

    /// Invalid parameter value
    #[error("Invalid parameter '{name}': {message}")]
    InvalidParameter {
        /// Parameter name
        name: String,
        /// Error message
        message: String,
    },

    /// Not implemented
    #[error("Not implemented: {0}")]
    NotImplemented(String),

    /// Other error
    #[error("Error: {0}")]
    Other(String),

    /// Core error
    #[error("Core error: {0}")]
    CoreError(#[from] CoreError),
}

/// Result type for time series operations
pub type Result<T> = std::result::Result<T, TimeSeriesError>;
