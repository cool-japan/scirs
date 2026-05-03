//! Proximal Optimization Methods
//!
//! This module provides proximal operators and splitting methods for optimising
//! non-smooth convex functions. These methods are particularly powerful for
//! regularised learning problems (LASSO, ridge, nuclear norm, etc.) and
//! image processing (total variation, sparsity-promoting penalties).
//!
//! # Structure
//!
//! | Submodule | Contents |
//! |-----------|----------|
//! | `operators` | `prox_l1`, `prox_l2`, `prox_linf`, `prox_nuclear`, `project_simplex`, `project_box` |
//! | `ista` | `IstaOptimizer`, `FistaOptimizer`, `ista_minimize`, `fista_minimize` |
//! | `admm` | `AdmmSolver`, `solve_lasso`, `solve_consensus` |
//! | `splitting` | `douglas_rachford`, `peaceman_rachford`, `forward_backward`, `primal_dual_chambolle_pock` |
//!
//! # Quick Start
//!
//! ## LASSO via FISTA
//! ```rust,no_run
//! use scirs2_optimize::proximal::{fista_minimize, prox_l1};
//!
//! let f = |x: &[f64]| 0.5 * x.iter().map(|&xi| xi * xi).sum::<f64>();
//! let grad_f = |x: &[f64]| x.to_vec();
//! let prox = |v: &[f64]| prox_l1(v, 0.1);
//!
//! let result = fista_minimize(f, grad_f, prox, vec![2.0, -3.0], 0.5, 500)
//!     .expect("FISTA failed");
//! ```
//!
//! ## ADMM LASSO
//! ```rust,no_run
//! use scirs2_optimize::proximal::solve_lasso;
//!
//! let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
//! let b = vec![1.5, -0.5];
//! let x = solve_lasso(&a, &b, 0.1).expect("LASSO failed");
//! ```
//!
//! ## Douglas-Rachford Splitting
//! ```rust,no_run
//! use scirs2_optimize::proximal::{douglas_rachford, prox_l1, prox_l2};
//!
//! let prox_f = |v: &[f64]| prox_l1(v, 0.5);
//! let prox_g = |v: &[f64]| prox_l2(v, 0.5);
//! let x = douglas_rachford(&prox_f, &prox_g, vec![2.0, -1.0], 1.0, 500);
//! ```

pub mod admm;
pub mod ista;
pub mod operators;
pub mod splitting;

// ─── Re-exports ──────────────────────────────────────────────────────────────

// Proximal operators
pub use operators::{project_box, project_simplex, prox_l1, prox_l2, prox_linf, prox_nuclear};

// ISTA / FISTA
pub use ista::{fista_minimize, ista_minimize, FistaOptimizer, IstaOptimizer, ProxOptResult};

// ADMM
pub use admm::{solve_consensus, solve_lasso, AdmmSolver};

// Splitting methods
pub use splitting::{
    douglas_rachford, douglas_rachford_tracked, dr_split, forward_backward, peaceman_rachford,
    primal_dual_chambolle_pock, DRResult, SplittingResult,
};
