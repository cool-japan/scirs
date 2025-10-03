// Error types for the SciRS2 signal processing module

use thiserror::Error;

#[allow(unused_imports)]
/// Signal processing error type
#[derive(Error, Debug)]
pub enum SignalError {
    /// Computation error
    #[error("Computation error: {0}")]
    ComputationError(String),

    /// Dimension mismatch error
    #[error("Dimension mismatch error: {0}")]
    DimensionMismatch(String),

    /// Value error (invalid value)
    #[error("Value error: {0}")]
    ValueError(String),

    /// Invalid argument error
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    /// Not implemented error
    #[error("Not implemented: {0}")]
    NotImplemented(String),

    /// Runtime error
    #[error("Runtime error: {0}")]
    RuntimeError(String),

    /// Shape mismatch error
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),
}

// Conversion from scirs2_core errors
impl From<scirs2_core::CoreError> for SignalError {
    fn from(err: scirs2_core::CoreError) -> Self {
        SignalError::ComputationError(format!("Core error: {err}"))
    }
}

// Conversion from FFT errors
impl From<scirs2_fft::FFTError> for SignalError {
    fn from(err: scirs2_fft::FFTError) -> Self {
        SignalError::ComputationError(format!("FFT error: {err}"))
    }
}

// Conversion from ndarray shape errors
impl From<scirs2_core::ndarray::ShapeError> for SignalError {
    fn from(err: scirs2_core::ndarray::ShapeError) -> Self {
        SignalError::ShapeMismatch(format!("Shape error: {err}"))
    }
}

// Conversion from std::io::Error
impl From<std::io::Error> for SignalError {
    fn from(err: std::io::Error) -> Self {
        SignalError::ComputationError(format!("IO error: {err}"))
    }
}

// Conversion from scirs2_linalg errors
// impl From<scirs2_linalg::LinalgError> for SignalError {
//     fn from(err: scirs2, linalg: LinalgError) -> Self {
//         SignalError::ComputationError(format!("Linear algebra error: {}", err))
//     }
// }

/// Result type for signal processing operations
pub type SignalResult<T> = Result<T, SignalError>;
