//! Error types for the autograd module
//!
//! This module defines the error types used throughout the autograd module.

use crate::ndarray;
use thiserror::Error;

/// Error type for autograd operations
#[derive(Debug, Clone, PartialEq, Error)]
pub enum OpError {
    /// Error related to ndarray operations
    #[error("{0}: {1}")]
    NdArrayError(String, scirs2_core::ndarray::ShapeError),

    /// Shape incompatibility error
    #[error("Incompatible shape: {0}")]
    IncompatibleShape(String),

    /// Unsupported type error
    #[error("Type unsupported: {0}")]
    TypeUnsupported(String),

    /// Invalid dimensions error
    #[error("Invalid dimensions: {0}")]
    InvalidDims(String),

    /// Index out of bounds error
    #[error("Out of bounds: {0}")]
    OutOfBounds(String),

    /// Runtime error
    #[error("Runtime error: {0}")]
    RuntimeError(String),

    /// Other error
    #[error("Error: {0}")]
    Other(String),
}

/// Error during tensor's evaluation.
#[derive(Debug, PartialEq, Clone, Error)]
pub enum EvalError {
    /// Error during `Op`'s computation.
    #[error("{0}")]
    OpError(#[from] OpError),

    /// Error related to variable access
    #[error("Variable error: {0}")]
    VariableError(String),

    /// Other evaluation error
    #[error("Evaluation error: {0}")]
    Other(String),
}

/// Generic error type for autograd operations
#[derive(Debug, Error)]
pub enum AutogradError {
    /// Operation error
    #[error("{0}")]
    OperationError(String),

    /// Shape mismatch error
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),

    /// Variable error
    #[error("Variable error: {0}")]
    VariableError(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Integration error with other SciRS2 modules
    #[error("Integration error: {0}")]
    IntegrationError(String),

    /// IO error
    #[error("IO error: {0}")]
    IOError(#[from] std::io::Error),

    /// OpError
    #[error("{0}")]
    OpError(#[from] OpError),

    /// EvalError
    #[error("{0}")]
    EvalError(#[from] EvalError),
}

/// Result type for autograd operations
pub type Result<T> = std::result::Result<T, AutogradError>;

/// Result type for evaluation operations
pub type EvalResult<T> = std::result::Result<T, EvalError>;

/// Result type for op operations
pub type OpResult<T> = std::result::Result<T, OpError>;
