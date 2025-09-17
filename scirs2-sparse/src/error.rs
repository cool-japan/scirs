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

    /// Convergence error for iterative algorithms
    #[error("Convergence error: {0}")]
    ConvergenceError(String),

    /// Index cast overflow error
    #[error("Index value {value} cannot be represented in the target type {target_type}")]
    IndexCastOverflow {
        value: usize,
        target_type: &'static str,
    },

    /// Invalid format error
    #[error("Invalid format: {0}")]
    InvalidFormat(String),

    /// I/O error
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// GPU error
    #[error("GPU error: {0}")]
    GpuError(#[from] crate::gpu_ops::GpuError),

    /// Compression error
    #[error("Compression error: {0}")]
    CompressionError(String),

    /// I/O error with custom message
    #[error("I/O error: {0}")]
    Io(String),

    /// Block not found error
    #[error("Block not found: {0}")]
    BlockNotFound(String),
}

/// Result type for sparse matrix/array operations
pub type SparseResult<T> = Result<T, SparseError>;

impl SparseError {
    /// Get a user-friendly description of the error with possible solutions
    pub fn help_message(&self) -> &'static str {
        match self {
            SparseError::DimensionMismatch { .. } => {
                "Ensure that vectors and matrices have compatible dimensions for the operation. \
                 For matrix-vector multiplication A*x, A must have the same number of columns as x has elements."
            },
            SparseError::IndexOutOfBounds { .. } => {
                "Check that all row and column indices are within the matrix bounds. \
                 Remember that indices are 0-based in Rust."
            },
            SparseError::ShapeMismatch { .. } => {
                "Verify that matrix shapes are compatible for the operation. \
                 For addition/subtraction, matrices must have identical shapes. \
                 For multiplication A*B, A's column count must equal B's row count."
            },
            SparseError::SingularMatrix(_) => {
                "The matrix is singular (non-invertible). This can happen when:\n\
                 - The matrix has zero or very small diagonal elements\n\
                 - Rows or columns are linearly dependent\n\
                 - The condition number is too large\n\
                 Consider using iterative methods or regularization."
            },
            SparseError::IterativeSolverFailure(_) => {
                "The iterative solver failed to converge. Try:\n\
                 - Increasing the maximum number of iterations\n\
                 - Using a better preconditioner\n\
                 - Reducing the convergence tolerance\n\
                 - Using a different solver method"
            },
            SparseError::ConvergenceError(_) => {
                "Algorithm failed to converge within the specified tolerance and iterations. \
                 Consider relaxing the tolerance or increasing iteration limits."
            },
            SparseError::InconsistentData { .. } => {
                "The sparse matrix data is inconsistent. Check that:\n\
                 - indptr array has correct length (nrows + 1)\n\
                 - indices and data arrays have the same length\n\
                 - indptr values are non-decreasing\n\
                 - All indices are within bounds"
            },
            SparseError::IndexCastOverflow { .. } => {
                "Index values are too large for the target type. \
                 Consider using a larger index type (e.g., usize instead of u32)."
            },
            SparseError::InvalidFormat(_) => {
                "The sparse matrix format is invalid or unsupported for this operation. \
                 Check that the matrix format matches the expected format for the operation."
            }_ => "Refer to the documentation for more information about this error."
        }
    }

    /// Get suggestions for how to fix this error
    pub fn suggestions(&self) -> Vec<&'static str> {
        match self {
            SparseError::DimensionMismatch { .. } => vec![
                "Check matrix and vector dimensions before operations",
                "Use .shape() to inspect matrix dimensions",
                "Consider transposing matrices if needed",
            ],
            SparseError::SingularMatrix(_) => vec![
                "Check matrix condition number with condest()",
                "Add regularization to improve conditioning",
                "Use iterative methods like CG or BiCGSTAB",
                "Try incomplete factorizations (ILU, IC)",
            ],
            SparseError::IterativeSolverFailure(_) => vec![
                "Increase max_iter in solver options",
                "Use preconditioning (Jacobi, SSOR, ILU)",
                "Try a different solver (GMRES, BiCGSTAB)",
                "Check matrix properties (symmetry, definiteness)",
            ],
            SparseError::InconsistentData { .. } => vec![
                "Use from_triplets() for safer construction",
                "Validate data with .check_format() if available",
                "Ensure indices are sorted when required",
            ],
            SparseError::InvalidFormat(_) => vec![
                "Convert matrix to the required format",
                "Use .to_csr() or .to_csc() for format conversion",
                "Check if the operation supports the current format",
            ],
            _ => vec!["Check the documentation for this operation"],
        }
    }

    /// Create a dimension mismatch error with helpful context
    pub fn dimension_mismatch_with_context(expected: usize, found: usize, operation: &str) -> Self {
        SparseError::DimensionMismatch { expected, found }
    }

    /// Create a shape mismatch error with helpful context
    pub fn shape_mismatch_with_context(
        expected: (usize, usize),
        found: (usize, usize),
        _operation: &str,
    ) -> Self {
        SparseError::ShapeMismatch { expected, found }
    }
}

impl From<String> for SparseError {
    fn from(error: String) -> Self {
        SparseError::ComputationError(error)
    }
}
