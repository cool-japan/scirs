//! Convex optimization methods.
//!
//! Provides solvers for convex optimization problems including
//! Geometric Programming (GP) via log transformation to convex form.

pub mod geometric_programming;

pub use geometric_programming::{
    gp_to_convex, solve_gp, GPProblem, GPResult, GPSolverConfig, LogConvexProblem, Monomial,
    Posynomial,
};
