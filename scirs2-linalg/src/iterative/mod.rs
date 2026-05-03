//! Krylov subspace iterative solvers for dense linear systems
//!
//! This module complements the `scirs2-sparse` Krylov solvers by providing
//! dense-matrix variants of the classical Krylov methods. All solvers accept
//! full `Array2<F>` matrices and support preconditioning where applicable.
//!
//! ## Algorithms
//!
//! | Solver | Target system | Guarantees |
//! |--------|---------------|------------|
//! | [`conjugate_gradient`] | Symmetric positive definite | Monotone energy-norm decrease |
//! | [`gmres`] | General (restarted) | Minimum residual norm over Krylov space |
//! | [`bicgstab`] | General non-symmetric | Smooth short-recurrence convergence |
//! | [`minres`] | Symmetric (possibly indefinite) | Minimum 2-norm residual |
//!
//! ## Result type
//!
//! All solvers return [`IterativeSolveResult`] which bundles the solution
//! vector with convergence diagnostics (iteration count, residual norm, and a
//! boolean convergence flag).

pub mod krylov;

pub use krylov::{bicgstab, conjugate_gradient, gmres, minres, IterativeSolveResult};
