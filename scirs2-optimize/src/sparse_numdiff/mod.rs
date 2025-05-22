//! Sparse numerical differentiation for large-scale optimization
//!
//! This module provides functions for computing sparse Jacobians and Hessians
//! using finite differences, designed for large-scale optimization problems.

// Module declarations
pub mod coloring;
pub mod compression;
pub mod finite_diff;
pub mod hessian;
pub mod jacobian;

// Re-exports for backward compatibility
pub use self::finite_diff::SparseFiniteDiffOptions;
pub use self::hessian::sparse_hessian;
pub use self::jacobian::sparse_jacobian;
