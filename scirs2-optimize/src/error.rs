//! Error types for the SciRS2 optimization module

use thiserror::Error;

/// Optimization error type
#[derive(Error, Debug)]
pub enum OptimizeError {
    /// Computation error (generic error)
    #[error("Computation error: {0}")]
    ComputationError(String),

    /// Convergence error (algorithm did not converge)
    #[error("Convergence error: {0}")]
    ConvergenceError(String),

    /// Value error (invalid value)
    #[error("Value error: {0}")]
    ValueError(String),

    /// Not implemented error
    #[error("Not implemented: {0}")]
    NotImplementedError(String),
}

/// Result type for optimization operations
pub type OptimizeResult<T> = Result<T, OptimizeError>;

// Implement conversion from SparseError to OptimizeError
impl From<scirs2_sparse::error::SparseError> for OptimizeError {
    fn from(error: scirs2_sparse::error::SparseError) -> Self {
        match error {
            scirs2_sparse::error::SparseError::ComputationError(msg) => {
                OptimizeError::ComputationError(msg)
            }
            scirs2_sparse::error::SparseError::DimensionMismatch { expected, found } => {
                OptimizeError::ValueError(format!(
                    "Dimension mismatch: expected {}, found {}",
                    expected, found
                ))
            }
            scirs2_sparse::error::SparseError::IndexOutOfBounds { index, shape } => {
                OptimizeError::ValueError(format!(
                    "Index {:?} out of bounds for array with shape {:?}",
                    index, shape
                ))
            }
            scirs2_sparse::error::SparseError::InvalidAxis => {
                OptimizeError::ValueError("Invalid axis specified".to_string())
            }
            scirs2_sparse::error::SparseError::InvalidSliceRange => {
                OptimizeError::ValueError("Invalid slice range specified".to_string())
            }
            scirs2_sparse::error::SparseError::InconsistentData { reason } => {
                OptimizeError::ValueError(format!("Inconsistent data: {}", reason))
            }
            scirs2_sparse::error::SparseError::NotImplemented(msg) => {
                OptimizeError::NotImplementedError(msg)
            }
            scirs2_sparse::error::SparseError::SingularMatrix(msg) => {
                OptimizeError::ComputationError(format!("Singular matrix error: {}", msg))
            }
            scirs2_sparse::error::SparseError::ValueError(msg) => OptimizeError::ValueError(msg),
            scirs2_sparse::error::SparseError::ConversionError(msg) => {
                OptimizeError::ValueError(format!("Conversion error: {}", msg))
            }
            scirs2_sparse::error::SparseError::OperationNotSupported(msg) => {
                OptimizeError::NotImplementedError(format!("Operation not supported: {}", msg))
            }
            scirs2_sparse::error::SparseError::ShapeMismatch { expected, found } => {
                OptimizeError::ValueError(format!(
                    "Shape mismatch: expected {:?}, found {:?}",
                    expected, found
                ))
            }
            scirs2_sparse::error::SparseError::IterativeSolverFailure(msg) => {
                OptimizeError::ConvergenceError(format!("Iterative solver failure: {}", msg))
            }
            scirs2_sparse::error::SparseError::IndexCastOverflow { value, target_type } => {
                OptimizeError::ValueError(format!(
                    "Index value {} cannot be represented in the target type {}",
                    value, target_type
                ))
            }
        }
    }
}
