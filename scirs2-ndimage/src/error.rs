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

    /// Core error (propagated from scirs2-core)
    #[error("{0}")]
    CoreError(#[from] CoreError),

    /// Other error
    #[error("Error: {0}")]
    Other(String),
}

// The #[from] attribute in the CoreError variant handles the conversion automatically

/// Result type for ndimage operations
pub type Result<T> = std::result::Result<T, NdimageError>;
