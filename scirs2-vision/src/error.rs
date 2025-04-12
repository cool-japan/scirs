//! Error types for the vision module

use thiserror::Error;

/// Vision module error type
#[derive(Error, Debug)]
pub enum VisionError {
    /// Image loading error
    #[error("Failed to load image: {0}")]
    ImageLoadError(String),

    /// Invalid parameter error
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Operation error
    #[error("Operation failed: {0}")]
    OperationError(String),

    /// Underlying ndimage error (temporarily simplified for publishing)
    #[error("ndimage error: {0}")]
    NdimageError(String),

    /// I/O error
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Type conversion error
    #[error("Type conversion error: {0}")]
    TypeConversionError(String),
}

/// Result type for vision operations
pub type Result<T> = std::result::Result<T, VisionError>;
