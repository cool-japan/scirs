//! Utility functions for DAE solvers
//!
//! This module provides utility functions for use in DAE solvers.

// Linear solvers
pub mod linear_solvers;

// Re-export useful utilities
pub use linear_solvers::{solve_linear_system, solve_lu, vector_norm, matrix_norm};