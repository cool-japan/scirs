//! Differential Algebraic Equation (DAE) solvers
//!
//! This module provides numerical solvers for differential algebraic equations (DAEs).
//! DAEs are a more general class of equations than ODEs and include algebraic constraints.
//!
//! Features:
//! - Support for index-1 DAE systems
//! - Semi-explicit DAE solvers for specialized forms
//! - Implicit DAE solvers for general forms
//! - Integration with mass matrix functionality
//! - Specialized BDF methods for DAE systems
//!
//! DAEs can be classified based on their index, which roughly corresponds to the
//! number of times one must differentiate the constraint equations to obtain an ODE.
//! This module focuses primarily on index-1 DAEs, which are the most common in practice.
//!
//! ## Semi-explicit Form
//!
//! Semi-explicit DAEs have the form:
//! ```text
//! x' = f(x, y, t)
//! 0 = g(x, y, t)
//! ```
//! where x are differential variables and y are algebraic variables.
//!
//! ## Fully-implicit Form
//!
//! Fully-implicit DAEs have the form:
//! ```text
//! F(t, y, y') = 0
//! ```
//! where all variables are treated uniformly and the system equations
//! depend on both the variables and their derivatives.

// Public types module
pub mod types;

// Public modules
pub mod index_reduction;
pub mod methods;
pub mod solvers;
pub mod utils;

// Re-export core types
pub use self::types::{DAEIndex, DAEOptions, DAEResult, DAEType};

// Re-export solver functions
pub use self::solvers::{
    solve_higher_index_dae, solve_implicit_dae, solve_ivp_dae, solve_semi_explicit_dae,
};

// Re-export specialized method functions
pub use self::methods::bdf_dae::{bdf_implicit_dae, bdf_semi_explicit_dae};
pub use self::methods::index_reduction_bdf::{
    bdf_implicit_with_index_reduction, bdf_with_index_reduction,
};
pub use self::methods::krylov_dae::{krylov_bdf_implicit_dae, krylov_bdf_semi_explicit_dae};

// Re-export preconditioner functions
pub use self::methods::block_precond::{
    create_block_ilu_preconditioner, create_block_jacobi_preconditioner,
};

// Re-export index reduction types
pub use self::index_reduction::{
    DAEStructure, DummyDerivativeReducer, PantelidesReducer, ProjectionMethod,
};
