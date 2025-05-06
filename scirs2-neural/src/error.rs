//! Error types for the neural network module

use std::error;
use std::fmt;

// Re-export Error trait for public use
pub use std::error::Error;

/// Error type for neural network operations
#[derive(Debug)]
pub enum NeuralError {
    /// Invalid architecture
    InvalidArchitecture(String),
    /// Training error
    TrainingError(String),
    /// Inference error
    InferenceError(String),
    /// Other error
    Other(String),
}

impl fmt::Display for NeuralError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            NeuralError::InvalidArchitecture(msg) => write!(f, "Invalid architecture: {}", msg),
            NeuralError::TrainingError(msg) => write!(f, "Training error: {}", msg),
            NeuralError::InferenceError(msg) => write!(f, "Inference error: {}", msg),
            NeuralError::Other(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl error::Error for NeuralError {}

/// Result type for neural network operations
pub type Result<T> = std::result::Result<T, NeuralError>;
