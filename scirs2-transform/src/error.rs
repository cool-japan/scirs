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

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Other error
    #[error("Error: {0}")]
    Other(String),
}

/// Result type for data transformation operations
pub type Result<T> = std::result::Result<T, TransformError>;
