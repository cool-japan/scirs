//! Error types for the ML optimization module

use std::error::Error;
use std::fmt;

/// Error type for ML optimization operations
#[derive(Debug)]
pub enum OptimError {
    /// Invalid configuration
    InvalidConfig(String),
    /// Optimization error
    OptimizationError(String),
    /// Dimension mismatch error
    DimensionMismatch(String),
    /// Other error
    Other(String),
}

impl fmt::Display for OptimError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            OptimError::InvalidConfig(msg) => write!(f, "Invalid configuration: {}", msg),
            OptimError::OptimizationError(msg) => write!(f, "Optimization error: {}", msg),
            OptimError::DimensionMismatch(msg) => write!(f, "Dimension mismatch: {}", msg),
            OptimError::Other(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl Error for OptimError {}

/// Result type for ML optimization operations
pub type Result<T> = std::result::Result<T, OptimError>;
