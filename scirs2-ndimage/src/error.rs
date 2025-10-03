//! Error types for the ndimage module
//!
//! This module provides error types for the ndimage module functions.

use scirs2_core::error::CoreError;
use thiserror::Error;

/// Error type for ndimage operations
#[derive(Error, Debug)]
pub enum NdimageError {
    /// Invalid input data
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Dimension error
    #[error("Dimension mismatch: {0}")]
    DimensionError(String),

    /// Implementation error
    #[error("Implementation error: {0}")]
    ImplementationError(String),

    /// Computation error
    #[error("Computation error: {0}")]
    ComputationError(String),

    /// Filter error
    #[error("Filter error: {0}")]
    FilterError(String),

    /// Interpolation error
    #[error("Interpolation error: {0}")]
    InterpolationError(String),

    /// Measurement error
    #[error("Measurement error: {0}")]
    MeasurementError(String),

    /// Morphology error
    #[error("Morphology error: {0}")]
    MorphologyError(String),

    /// Not implemented error
    #[error("Not implemented: {0}")]
    NotImplementedError(String),

    /// Processing error
    #[error("Processing error: {0}")]
    ProcessingError(String),

    /// Core error (propagated from scirs2-core)
    #[error("{0}")]
    CoreError(#[from] CoreError),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Shape building error
    #[error("Shape error: {0}")]
    ShapeError(#[from] scirs2_core::ndarray::ShapeError),

    /// Format error
    #[error("Format error: {0}")]
    FormatError(#[from] std::fmt::Error),

    /// GPU not available error
    #[error("GPU not available: {0}")]
    GpuNotAvailable(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    /// Memory error
    #[error("Memory error: {0}")]
    MemoryError(String),

    /// Other error
    #[error("Error: {0}")]
    Other(String),
}

// The #[from] attribute in the CoreError variant handles the conversion automatically

/// Result type for ndimage operations
pub type NdimageResult<T> = std::result::Result<T, NdimageError>;
