//! Methods for solving Differential Algebraic Equations (DAEs)
//!
//! This module provides the implementation of numerical methods for solving DAE systems.
//! The methods are organized based on the type of DAE system they are designed to solve:
//! - Backward Differentiation Formula (BDF) methods for stiff DAEs
//! - Specialized methods for semi-explicit DAEs
//! - Specialized methods for fully implicit DAEs
//! - Index reduction techniques combined with BDF methods for higher-index DAEs
//! - Krylov subspace methods for large DAE systems
//! - Block-structured preconditioners for improving Krylov method performance

// BDF methods for DAE systems
pub mod bdf_dae;

// Index reduction BDF methods for higher-index DAEs
pub mod index_reduction_bdf;

// Krylov subspace methods for large DAE systems
pub mod krylov_dae;

// Block-structured preconditioners for DAE systems
pub mod block_precond;

// Re-export main solver functions
pub use self::bdf_dae::{bdf_implicit_dae, bdf_semi_explicit_dae};
pub use self::index_reduction_bdf::{bdf_implicit_with_index_reduction, bdf_with_index_reduction};
pub use self::krylov_dae::{krylov_bdf_implicit_dae, krylov_bdf_semi_explicit_dae};

// Re-export preconditioner creation functions
pub use self::block_precond::{
    create_block_ilu_preconditioner, create_block_jacobi_preconditioner,
};
