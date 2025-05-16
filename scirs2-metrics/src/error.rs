//! Error types for the metrics module

use thiserror::Error;

/// Error type for metrics operations
#[derive(Error, Debug)]
pub enum MetricsError {
    /// Invalid input data
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Calculation error
    #[error("Calculation error: {0}")]
    CalculationError(String),

    /// Computation error (used in parallel and streaming contexts)
    #[error("Computation error: {0}")]
    ComputationError(String),

    /// Dimension mismatch between arrays
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// Shape mismatch between arrays
    #[error("Shape mismatch: shape1 = {shape1}, shape2 = {shape2}")]
    ShapeMismatch {
        /// First shape in mismatch
        shape1: String,
        /// Second shape in mismatch
        shape2: String,
    },

    /// Invalid argument
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    /// Division by zero error
    #[error("Division by zero")]
    DivisionByZero,

    /// Statistics error
    #[error("Statistics error: {0}")]
    StatsError(String),

    /// Linear algebra error
    #[error("Linear algebra error: {0}")]
    LinalgError(String),

    /// IO error
    #[error("IO error: {0}")]
    IOError(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Deserialization error
    #[error("Deserialization error: {0}")]
    DeserializationError(String),

    /// Visualization error
    #[error("Visualization error: {0}")]
    VisualizationError(String),

    /// Core error
    #[error("Core error: {0}")]
    CoreError(#[from] scirs2_core::error::CoreError),

    /// Other error
    #[error("Error: {0}")]
    Other(String),
}

/// Result type for metrics operations
pub type Result<T> = std::result::Result<T, MetricsError>;
