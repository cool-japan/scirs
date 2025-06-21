//! Utility functions for ODE solvers.

pub mod common;
pub mod dense_output;
pub mod diagnostics;
pub mod events;
pub mod interpolation;
pub mod jacobian;
pub mod linear_solvers;
pub mod mass_matrix;
#[cfg(feature = "simd")]
pub mod simd_ops;
pub mod step_control;
pub mod stiffness;

// Re-exports
pub use dense_output::*;
pub use diagnostics::*;
pub use interpolation::*;
pub use jacobian::*;
pub use linear_solvers::*;
pub use step_control::*;
pub use stiffness::*;

// Selective imports from common to avoid conflicts
pub use common::{
    calculate_error_weights, estimate_initial_step, extrapolate, finite_difference_jacobian,
    scaled_norm,
};

// SIMD operations (feature-gated)
#[cfg(feature = "simd")]
pub use simd_ops::SimdOdeOps;

// Don't re-export events or mass_matrix as they have potential naming conflicts
