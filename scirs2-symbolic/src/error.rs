//! Error types for symbolic computation.

use thiserror::Error;

/// Errors that can occur during symbolic expression evaluation or manipulation.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum SymbolicError {
    /// A variable referenced in the expression has no binding in the evaluation context.
    #[error("Unbound variable '{0}'")]
    UnboundVariable(String),

    /// Division by zero encountered during evaluation.
    #[error("Division by zero in symbolic evaluation")]
    DivisionByZero,

    /// A mathematical domain violation (e.g., `ln` of a non-positive number).
    #[error("Domain error: {0}")]
    DomainError(String),

    /// An expression contains a structural cycle or is pathologically deep.
    #[error("Expression structure error: {0}")]
    StructureError(String),
}
