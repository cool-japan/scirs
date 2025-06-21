//! Error types for the clustering module

use scirs2_core::error::CoreError;
use scirs2_linalg::error::LinalgError;
use thiserror::Error;

/// Error type for clustering operations
#[derive(Error, Debug)]
pub enum ClusteringError {
    /// Invalid input data
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Computation error
    #[error("Computation error: {0}")]
    ComputationError(String),

    /// Vector quantization specific errors
    #[error("Vector quantization error: {0}")]
    VqError(String),

    /// Hierarchical clustering specific errors
    #[error("Hierarchical clustering error: {0}")]
    HierarchyError(String),

    /// Empty cluster error
    #[error("Empty cluster error: {0}")]
    EmptyCluster(String),

    /// Invalid state error
    #[error("Invalid state: {0}")]
    InvalidState(String),

    /// Linear algebra error
    #[error("Linear algebra error: {0}")]
    LinalgError(#[from] LinalgError),

    /// Core validation error
    #[error("Core validation error: {0}")]
    CoreError(#[from] CoreError),

    /// Other error
    #[error("Error: {0}")]
    Other(String),
}

/// Result type for clustering operations
pub type Result<T> = std::result::Result<T, ClusteringError>;
