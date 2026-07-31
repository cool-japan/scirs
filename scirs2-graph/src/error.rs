//! Error types for the graph processing module
//!
//! This module provides comprehensive error handling for graph operations,
//! including detailed context information and recovery suggestions.

use std::fmt;
use thiserror::Error;

/// Error type for graph processing operations
///
/// Provides detailed error information with context and suggestions for recovery.
/// All errors include location information when possible.
#[derive(Error, Debug)]
pub enum GraphError {
    /// Node not found in the graph
    #[error("Node {node} not found in graph with {graph_size} nodes. Context: {context}")]
    NodeNotFound {
        /// The node that was not found
        node: String,
        /// Size of the graph for context
        graph_size: usize,
        /// Additional context about the operation
        context: String,
    },

    /// Edge not found in the graph
    #[error("Edge ({src_node}, {target}) not found in graph. Context: {context}")]
    EdgeNotFound {
        /// Source node of the edge
        src_node: String,
        /// Target node of the edge
        target: String,
        /// Additional context about the operation
        context: String,
    },

    /// Invalid parameter provided to an operation
    #[error("Invalid parameter '{param}' with value '{value}'. Expected: {expected}. Context: {context}")]
    InvalidParameter {
        /// Parameter name
        param: String,
        /// Provided value
        value: String,
        /// Expected value or range
        expected: String,
        /// Additional context
        context: String,
    },

    /// Algorithm failed to converge or complete
    #[error("Algorithm '{algorithm}' failed: {reason}. Iterations: {iterations}, Tolerance: {tolerance}")]
    AlgorithmFailure {
        /// Name of the algorithm
        algorithm: String,
        /// Reason for failure
        reason: String,
        /// Number of iterations completed
        iterations: usize,
        /// Tolerance used
        tolerance: f64,
    },

    /// I/O operation failed
    #[error("I/O error for path '{path}': {source}")]
    IOError {
        /// File path that caused the error
        path: String,
        /// Underlying I/O error
        #[source]
        source: std::io::Error,
    },

    /// Memory allocation or usage error
    #[error("Memory error: requested {requested} bytes, available {available} bytes. Context: {context}")]
    MemoryError {
        /// Requested memory in bytes
        requested: usize,
        /// Available memory in bytes
        available: usize,
        /// Additional context
        context: String,
    },

    /// Algorithm did not converge within specified limits
    #[error("Convergence error in '{algorithm}': completed {iterations} iterations with tolerance {tolerance}, threshold {threshold}")]
    ConvergenceError {
        /// Algorithm name
        algorithm: String,
        /// Iterations completed
        iterations: usize,
        /// Final tolerance achieved
        tolerance: f64,
        /// Required threshold
        threshold: f64,
    },

    /// Graph structure is invalid for the operation
    #[error("Graph structure error: expected {expected}, found {found}. Context: {context}")]
    GraphStructureError {
        /// Expected graph property
        expected: String,
        /// Actual graph property
        found: String,
        /// Additional context
        context: String,
    },

    /// No path exists between nodes
    #[error(
        "No path found from {src_node} to {target} in graph with {nodes} nodes and {edges} edges"
    )]
    NoPath {
        /// Source node
        src_node: String,
        /// Target node
        target: String,
        /// Number of nodes in graph
        nodes: usize,
        /// Number of edges in graph
        edges: usize,
    },

    /// Cycle detected when acyclic graph expected
    #[error(
        "Cycle detected in graph starting from node {start_node}. Cycle length: {cycle_length}"
    )]
    CycleDetected {
        /// Node where cycle starts
        start_node: String,
        /// Length of the detected cycle
        cycle_length: usize,
    },

    /// Linear algebra operation failed
    #[error("Linear algebra error in operation '{operation}': {details}")]
    LinAlgError {
        /// Operation that failed
        operation: String,
        /// Error details
        details: String,
    },

    /// Sparse matrix operation failed
    #[error("Sparse matrix error: {details}")]
    SparseError {
        /// Error details
        details: String,
    },

    /// Core module error
    #[error("Core module error: {0}")]
    CoreError(#[from] scirs2_core::error::CoreError),

    /// Serialization/deserialization failed
    #[error("Serialization error for format '{format}': {details}")]
    SerializationError {
        /// Data format (JSON, OxiCode, etc.)
        format: String,
        /// Error details
        details: String,
    },

    /// Invalid graph attribute
    #[error("Invalid attribute '{attribute}' for {target_type}: {details}")]
    InvalidAttribute {
        /// Attribute name
        attribute: String,
        /// Target type (node, edge, graph)
        target_type: String,
        /// Error details
        details: String,
    },

    /// Computation was cancelled or interrupted
    #[error("Operation '{operation}' was cancelled after {elapsed_time} seconds")]
    Cancelled {
        /// Operation name
        operation: String,
        /// Time elapsed before cancellation
        elapsed_time: f64,
    },

    /// Thread safety or concurrency error
    #[error("Concurrency error in '{operation}': {details}")]
    ConcurrencyError {
        /// Operation name
        operation: String,
        /// Error details
        details: String,
    },

    /// Invalid graph format or version
    #[error("Format error: unsupported format '{format}' version {version}. Supported versions: {supported}")]
    FormatError {
        /// Format name
        format: String,
        /// Version found
        version: String,
        /// Supported versions
        supported: String,
    },

    /// Invalid graph structure (legacy error for backward compatibility)
    #[error("Invalid graph: {0}")]
    InvalidGraph(String),

    /// Algorithm error (legacy error for backward compatibility)
    #[error("Algorithm error: {0}")]
    AlgorithmError(String),

    /// Computation error (legacy error for backward compatibility)
    #[error("Computation error: {0}")]
    ComputationError(String),

    /// A requested capability is not implemented and, unlike most gaps in
    /// this crate, genuinely is not planned to be: the feature is out of
    /// scope for what a graph-processing library can honestly provide (e.g.
    /// it would require hardware or a simulator this crate does not have
    /// access to). Returned instead of silently no-op'ing or fabricating a
    /// plausible-looking result.
    #[error("Unsupported: {0}")]
    Unsupported(String),

    /// Generic error for backward compatibility
    #[error("{0}")]
    Other(String),
}

impl GraphError {
    /// Create a NodeNotFound error with minimal context
    pub fn node_not_found<T: fmt::Display>(node: T) -> Self {
        Self::NodeNotFound {
            node: node.to_string(),
            graph_size: 0,
            context: "Node lookup operation".to_string(),
        }
    }

    /// Create a NodeNotFound error with full context
    pub fn node_not_found_with_context<T: fmt::Display>(
        node: T,
        graph_size: usize,
        context: &str,
    ) -> Self {
        Self::NodeNotFound {
            node: node.to_string(),
            graph_size,
            context: context.to_string(),
        }
    }

    /// Create an EdgeNotFound error with minimal context
    pub fn edge_not_found<S: fmt::Display, T: fmt::Display>(source: S, target: T) -> Self {
        Self::EdgeNotFound {
            src_node: source.to_string(),
            target: target.to_string(),
            context: "Edge lookup operation".to_string(),
        }
    }

    /// Create an EdgeNotFound error with full context
    pub fn edge_not_found_with_context<S: fmt::Display, T: fmt::Display>(
        source: S,
        target: T,
        context: &str,
    ) -> Self {
        Self::EdgeNotFound {
            src_node: source.to_string(),
            target: target.to_string(),
            context: context.to_string(),
        }
    }

    /// Create an InvalidParameter error
    pub fn invalid_parameter<P: fmt::Display, V: fmt::Display, E: fmt::Display>(
        param: P,
        value: V,
        expected: E,
    ) -> Self {
        Self::InvalidParameter {
            param: param.to_string(),
            value: value.to_string(),
            expected: expected.to_string(),
            context: "Parameter validation".to_string(),
        }
    }

    /// Create an AlgorithmFailure error
    pub fn algorithm_failure<A: fmt::Display, R: fmt::Display>(
        algorithm: A,
        reason: R,
        iterations: usize,
        tolerance: f64,
    ) -> Self {
        Self::AlgorithmFailure {
            algorithm: algorithm.to_string(),
            reason: reason.to_string(),
            iterations,
            tolerance,
        }
    }

    /// Create a MemoryError
    pub fn memory_error(requested: usize, available: usize, context: &str) -> Self {
        Self::MemoryError {
            requested,
            available,
            context: context.to_string(),
        }
    }

    /// Create a ConvergenceError
    pub fn convergence_error<A: fmt::Display>(
        algorithm: A,
        iterations: usize,
        tolerance: f64,
        threshold: f64,
    ) -> Self {
        Self::ConvergenceError {
            algorithm: algorithm.to_string(),
            iterations,
            tolerance,
            threshold,
        }
    }

    /// Create a GraphStructureError
    pub fn graph_structure_error<E: fmt::Display, F: fmt::Display>(
        expected: E,
        found: F,
        context: &str,
    ) -> Self {
        Self::GraphStructureError {
            expected: expected.to_string(),
            found: found.to_string(),
            context: context.to_string(),
        }
    }

    /// Create a NoPath error
    pub fn no_path<S: fmt::Display, T: fmt::Display>(
        source: S,
        target: T,
        nodes: usize,
        edges: usize,
    ) -> Self {
        Self::NoPath {
            src_node: source.to_string(),
            target: target.to_string(),
            nodes,
            edges,
        }
    }

    /// Check if this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            GraphError::NodeNotFound { .. } => true,
            GraphError::EdgeNotFound { .. } => true,
            GraphError::NoPath { .. } => true,
            GraphError::InvalidParameter { .. } => true,
            GraphError::ConvergenceError { .. } => true,
            GraphError::Cancelled { .. } => true,
            GraphError::AlgorithmFailure { .. } => false,
            GraphError::GraphStructureError { .. } => false,
            GraphError::CycleDetected { .. } => false,
            GraphError::LinAlgError { .. } => false,
            GraphError::SparseError { .. } => false,
            GraphError::SerializationError { .. } => false,
            GraphError::InvalidAttribute { .. } => true,
            GraphError::ConcurrencyError { .. } => false,
            GraphError::FormatError { .. } => false,
            GraphError::InvalidGraph(_) => false,
            GraphError::AlgorithmError(_) => false,
            GraphError::MemoryError { .. } => false,
            GraphError::IOError { .. } => false,
            GraphError::CoreError(_) => false,
            GraphError::ComputationError(_) => false,
            GraphError::Unsupported(_) => false,
            GraphError::Other(_) => false,
        }
    }

    /// Get suggestions for error recovery
    pub fn recovery_suggestions(&self) -> Vec<String> {
        match self {
            GraphError::NodeNotFound { .. } => vec![
                "Check that the node exists in the graph".to_string(),
                "Verify node ID format and type".to_string(),
                "Use graph.has_node() to check existence first".to_string(),
            ],
            GraphError::EdgeNotFound { .. } => vec![
                "Check that both nodes exist in the graph".to_string(),
                "Verify edge direction for directed graphs".to_string(),
                "Use graph.has_edge() to check existence first".to_string(),
            ],
            GraphError::NoPath { .. } => vec![
                "Check if graph is connected".to_string(),
                "Verify that both nodes exist".to_string(),
                "Consider using weakly connected components for directed graphs".to_string(),
            ],
            GraphError::AlgorithmFailure { algorithm, .. } => match algorithm.as_str() {
                "pagerank" => vec![
                    "Increase iteration limit".to_string(),
                    "Reduce tolerance threshold".to_string(),
                    "Check for disconnected components".to_string(),
                ],
                "community_detection" => vec![
                    "Try different resolution parameters".to_string(),
                    "Ensure graph has edges".to_string(),
                    "Consider preprocessing to remove isolates".to_string(),
                ],
                _ => vec!["Adjust algorithm parameters".to_string()],
            },
            GraphError::MemoryError { .. } => vec![
                "Use streaming algorithms for large graphs".to_string(),
                "Enable memory optimization features".to_string(),
                "Process graph in smaller chunks".to_string(),
            ],
            GraphError::ConvergenceError { .. } => vec![
                "Increase maximum iterations".to_string(),
                "Adjust tolerance threshold".to_string(),
                "Check for numerical stability issues".to_string(),
            ],
            GraphError::Unsupported(_) => vec![
                "This capability is out of scope for this crate and is not planned; \
                 no retry or reconfiguration will make it succeed"
                    .to_string(),
                "Use a CPU-based (non-accelerated) code path instead".to_string(),
            ],
            _ => vec!["Check input parameters and graph structure".to_string()],
        }
    }

    /// Get the error category for metrics and logging
    pub fn category(&self) -> &'static str {
        match self {
            GraphError::NodeNotFound { .. } | GraphError::EdgeNotFound { .. } => "lookup",
            GraphError::InvalidParameter { .. } => "validation",
            GraphError::AlgorithmFailure { .. } | GraphError::ConvergenceError { .. } => {
                "algorithm"
            }
            GraphError::IOError { .. } => "io",
            GraphError::MemoryError { .. } => "memory",
            GraphError::GraphStructureError { .. } => "structure",
            GraphError::NoPath { .. } => "connectivity",
            GraphError::CycleDetected { .. } => "topology",
            GraphError::SerializationError { .. } => "serialization",
            GraphError::Cancelled { .. } => "cancellation",
            GraphError::ConcurrencyError { .. } => "concurrency",
            GraphError::FormatError { .. } => "format",
            GraphError::Unsupported(_) => "unsupported",
            _ => "other",
        }
    }
}

/// Result type for graph processing operations
pub type Result<T> = std::result::Result<T, GraphError>;

/// Convert std::io::Error to GraphError with path context
impl From<std::io::Error> for GraphError {
    fn from(err: std::io::Error) -> Self {
        GraphError::IOError {
            path: "unknown".to_string(),
            source: err,
        }
    }
}

/// Error context helper for adding operation context to errors
pub struct ErrorContext {
    operation: String,
    graph_info: Option<(usize, usize)>, // (nodes, edges)
}

impl ErrorContext {
    /// Create new error context
    pub fn new(operation: &str) -> Self {
        Self {
            operation: operation.to_string(),
            graph_info: None,
        }
    }

    /// Add graph size information
    pub fn with_graph_info(mut self, nodes: usize, edges: usize) -> Self {
        self.graph_info = Some((nodes, edges));
        self
    }

    /// Wrap a result with context information
    pub fn wrap<T>(self, result: Result<T>) -> Result<T> {
        result.map_err(|err| self.add_context(err))
    }

    /// Add context to an existing error
    fn add_context(self, mut err: GraphError) -> GraphError {
        match &mut err {
            GraphError::NodeNotFound { context, .. } if context == "Node lookup operation" => {
                *context = self.operation;
            }
            GraphError::EdgeNotFound { context, .. } if context == "Edge lookup operation" => {
                *context = self.operation;
            }
            GraphError::InvalidParameter { context, .. } if context == "Parameter validation" => {
                *context = self.operation;
            }
            GraphError::GraphStructureError { context, .. } => {
                *context = self.operation;
            }
            _ => {}
        }
        err
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = GraphError::node_not_found(42);
        assert!(matches!(err, GraphError::NodeNotFound { .. }));
        assert!(err.is_recoverable());
        assert_eq!(err.category(), "lookup");
    }

    #[test]
    fn test_error_context() {
        let _ctx = ErrorContext::new("PageRank computation").with_graph_info(100, 250);
        let err = GraphError::convergence_error("pagerank", 100, 1e-3, 1e-6);
        let suggestions = err.recovery_suggestions();
        assert!(!suggestions.is_empty());
    }

    #[test]
    fn test_error_categories() {
        assert_eq!(GraphError::node_not_found(1).category(), "lookup");
        assert_eq!(
            GraphError::algorithm_failure("test", "failed", 0, 1e-6).category(),
            "algorithm"
        );
        assert_eq!(
            GraphError::memory_error(1000, 500, "test").category(),
            "memory"
        );
    }
}
