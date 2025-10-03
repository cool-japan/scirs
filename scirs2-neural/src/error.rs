//! Error types for the neural network module

use std::error;
use std::fmt;
// Re-export Error trait for public use
pub use std::error::Error as StdError;
/// Error type for neural network operations
#[derive(Debug)]
pub enum NeuralError {
    /// Invalid architecture
    InvalidArchitecture(String),
    /// Training error
    TrainingError(String),
    /// Inference error
    InferenceError(String),
    /// Serialization error
    SerializationError(String),
    /// Deserialization error
    DeserializationError(String),
    /// Validation error
    ValidationError(String),
    /// Not implemented error
    NotImplementedError(String),
    /// IO error
    IOError(String),
    /// Invalid argument error
    InvalidArgument(String),
    /// Shape mismatch error  
    ShapeMismatch(String),
    /// Computation error
    ComputationError(String),
    /// Dimension mismatch error
    DimensionMismatch(String),
    /// Distributed training error
    DistributedError(String),
    /// Configuration error
    ConfigError(String),
    /// Allocation error
    AllocationError(String),
    /// Device error
    DeviceError(String),
    /// Device not found error
    DeviceNotFound(String),
    /// Resource exhausted error
    ResourceExhausted(String),
    /// Invalid state error
    InvalidState(String),
    /// Not implemented error (alternative name)
    NotImplemented(String),
    /// Memory error
    MemoryError(String),
    /// Feature not enabled error
    FeatureNotEnabled(String),
    /// Other error
    Other(String),
}
impl fmt::Display for NeuralError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            NeuralError::InvalidArchitecture(msg) => write!(f, "Invalid architecture: {msg}"),
            NeuralError::TrainingError(msg) => write!(f, "Training error: {msg}"),
            NeuralError::InferenceError(msg) => write!(f, "Inference error: {msg}"),
            NeuralError::SerializationError(msg) => write!(f, "Serialization error: {msg}"),
            NeuralError::DeserializationError(msg) => write!(f, "Deserialization error: {msg}"),
            NeuralError::ValidationError(msg) => write!(f, "Validation error: {msg}"),
            NeuralError::NotImplementedError(msg) => write!(f, "Not implemented: {msg}"),
            NeuralError::IOError(msg) => write!(f, "IO error: {msg}"),
            NeuralError::InvalidArgument(msg) => write!(f, "Invalid argument: {msg}"),
            NeuralError::ShapeMismatch(msg) => write!(f, "Shape mismatch: {msg}"),
            NeuralError::ComputationError(msg) => write!(f, "Computation error: {msg}"),
            NeuralError::DimensionMismatch(msg) => write!(f, "Dimension mismatch: {msg}"),
            NeuralError::DistributedError(msg) => write!(f, "Distributed training error: {msg}"),
            NeuralError::ConfigError(msg) => write!(f, "Configuration error: {msg}"),
            NeuralError::AllocationError(msg) => write!(f, "Allocation error: {msg}"),
            NeuralError::DeviceError(msg) => write!(f, "Device error: {msg}"),
            NeuralError::DeviceNotFound(msg) => write!(f, "Device not found: {msg}"),
            NeuralError::ResourceExhausted(msg) => write!(f, "Resource exhausted: {msg}"),
            NeuralError::InvalidState(msg) => write!(f, "Invalid state: {msg}"),
            NeuralError::NotImplemented(msg) => write!(f, "Not implemented: {msg}"),
            NeuralError::MemoryError(msg) => write!(f, "Memory error: {msg}"),
            NeuralError::FeatureNotEnabled(msg) => write!(f, "Feature not enabled: {msg}"),
            NeuralError::Other(msg) => write!(f, "Error: {msg}"),
        }
    }
}

impl error::Error for NeuralError {}
/// Error type alias
pub type Error = NeuralError;
/// Result type for neural network operations
pub type Result<T> = std::result::Result<T, Error>;
/// Dummy GPU backend type for compilation when GPU features are not available
#[cfg(not(feature = "gpu"))]
pub struct DummyGpuBackend;
// Implement conversion from std::io::Error to NeuralError
impl From<std::io::Error> for NeuralError {
    fn from(error: std::io::Error) -> Self {
        NeuralError::IOError(error.to_string())
    }
}

// Implement conversion from scirs2_core::ndarray::ShapeError to NeuralError
impl From<scirs2_core::ndarray::ShapeError> for NeuralError {
    fn from(error: scirs2_core::ndarray::ShapeError) -> Self {
        NeuralError::ShapeMismatch(format!("Shape error: {error}"))
    }
}
