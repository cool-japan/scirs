//! Symbolic attention mechanisms for SciRS2.
//!
//! This module provides symbolic wrappers for attention-related computations,
//! expressing them as [`crate::eml::LoweredOp`] expression trees for symbolic
//! manipulation, differentiation, and verification.
//!
//! # Modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`symbolic_alibi`] | Symbolic ALiBi (Attention with Linear Biases) slopes and bias matrices |

pub mod symbolic_alibi;

pub use symbolic_alibi::{
    alibi_bias_expr, alibi_bias_matrix_symbolic, alibi_slope, verify_symbolic_vs_numerical,
};
