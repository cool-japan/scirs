//! Ordinary Differential Equation solvers
//!
//! This module provides numerical solvers for ordinary differential equations (ODEs).
//! It includes a variety of methods for solving initial value problems (IVPs).
//!
//! Features:
//! - Multiple methods including explicit and implicit solvers
//! - Automatic stiffness detection and method switching
//! - Dense output for continuous solution approximation
//! - Event detection and handling for detecting specific conditions
//! - Support for different error control schemes

// Public types module
pub mod types;

// Public modules
pub mod methods;
pub mod solver;
pub mod utils;

// Re-export core types
pub use self::types::{MassMatrix, MassMatrixType, ODEMethod, ODEOptions, ODEResult};

// Re-export solver functions
pub use self::solver::{solve_ivp, solve_ivp_with_events};

// Re-export event detection types
pub use self::utils::events::{
    terminal_event, EventAction, EventDirection, EventSpec, ODEOptionsWithEvents,
    ODEResultWithEvents,
};
