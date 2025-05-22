//! Error types for the SciRS2 sparse module
//!
//! This module provides error types for both sparse matrix and sparse array operations.

use thiserror::Error;

/// Sparse matrix/array error type
#[derive(Error, Debug)]
pub enum SparseError {
    /// Computation error (generic error)
    #[error("Computation error: {0}")]
    ComputationError(String),

    /// Dimension mismatch error
    #[error("Dimension mismatch: expected {expected}, found {found}")]
    DimensionMismatch { expected: usize, found: usize },

    /// Index out of bounds error
    #[error("Index {index:?} out of bounds for array with shape {shape:?}")]
    IndexOutOfBounds {
        index: (usize, usize),
        shape: (usize, usize),
    },

    /// Invalid axis error
    #[error("Invalid axis specified")]
    InvalidAxis,

    /// Invalid slice range error
    #[error("Invalid slice range specified")]
    InvalidSliceRange,

    /// Inconsistent data error
    #[error("Inconsistent data: {reason}")]
    InconsistentData { reason: String },

    /// Not implemented error
    #[error("Feature not implemented: {0}")]
    NotImplemented(String),

    /// Singular matrix error
    #[error("Singular matrix error: {0}")]
    SingularMatrix(String),

    /// Value error (invalid value)
    #[error("Value error: {0}")]
    ValueError(String),

    /// Conversion error
    #[error("Conversion error: {0}")]
    ConversionError(String),

    /// Operation not supported error
    #[error("Operation not supported: {0}")]
    OperationNotSupported(String),

    /// Shape mismatch error
    #[error("Shape mismatch: expected {expected:?}, found {found:?}")]
    ShapeMismatch {
        expected: (usize, usize),
        found: (usize, usize),
    },

    /// Iterative solver failure error
    #[error("Iterative solver failure: {0}")]
    IterativeSolverFailure(String),

    /// Index cast overflow error
    #[error("Index value {value} cannot be represented in the target type {target_type}")]
    IndexCastOverflow {
        value: usize,
        target_type: &'static str,
    },
}

/// Result type for sparse matrix/array operations
pub type SparseResult<T> = Result<T, SparseError>;
