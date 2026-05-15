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

/// Errors specific to the EML substrate (Phase 0 of v0.4.4).
///
/// Distinct from [`SymbolicError`] so the legacy `Expr`-based API remains
/// stable while the EML-native pipeline grows its own error vocabulary.
#[derive(Debug, Clone, Error)]
pub enum EmlError {
    /// A variable index referenced in a `LoweredOp` is out of bounds.
    #[error("variable index {idx} out of bounds (have {len} bindings)")]
    UnboundVariableIndex {
        /// The offending index.
        idx: usize,
        /// The actual length of the variable binding vector.
        len: usize,
    },

    /// A variable name in an `Expr` was not found in the `VarMap`.
    #[error("unknown variable '{0}'")]
    UnknownVariable(String),

    /// A constant constructor received an invalid input (e.g. `Canonical::nat(0)`).
    #[error("invalid constant: {0}")]
    InvalidConstant(String),

    /// The parser failed to recognise input.
    #[error("parse error at position {position}: {message}")]
    ParseError {
        /// Byte offset where the parse failed.
        position: usize,
        /// Human-readable description of the failure.
        message: String,
    },

    /// Lowering an `EmlTree` to a `LoweredOp` failed (e.g. unrecognised
    /// canonical shape).
    #[error("lowering failed: {0}")]
    LoweringFailed(String),

    /// Numerical evaluation produced a non-finite or domain-violating result.
    #[error("evaluation domain error: {0}")]
    EvalDomain(String),

    /// A division-by-zero occurred during evaluation.
    #[error("division by zero")]
    DivisionByZero,
}
