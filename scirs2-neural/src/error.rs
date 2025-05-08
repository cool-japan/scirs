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
    NotImplemented(String),
    /// IO error
    IOError(String),
    /// Other error
    Other(String),
}

impl fmt::Display for NeuralError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            NeuralError::InvalidArchitecture(msg) => write!(f, "Invalid architecture: {}", msg),
            NeuralError::TrainingError(msg) => write!(f, "Training error: {}", msg),
            NeuralError::InferenceError(msg) => write!(f, "Inference error: {}", msg),
            NeuralError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            NeuralError::DeserializationError(msg) => write!(f, "Deserialization error: {}", msg),
            NeuralError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            NeuralError::NotImplemented(msg) => write!(f, "Not implemented: {}", msg),
            NeuralError::IOError(msg) => write!(f, "IO error: {}", msg),
            NeuralError::Other(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl error::Error for NeuralError {}

/// Error type alias
pub type Error = NeuralError;

/// Result type for neural network operations
pub type Result<T> = std::result::Result<T, Error>;

// Implement conversion from std::io::Error to NeuralError
impl From<std::io::Error> for NeuralError {
    fn from(error: std::io::Error) -> Self {
        NeuralError::IOError(error.to_string())
    }
}

// Implement conversion from ndarray::ShapeError to NeuralError
impl From<ndarray::ShapeError> for NeuralError {
    fn from(error: ndarray::ShapeError) -> Self {
        NeuralError::InferenceError(format!("Shape error: {}", error))
    }
}
