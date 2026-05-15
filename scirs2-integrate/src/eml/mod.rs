//! Symbolic EML integration: ODE drivers and quadrature with exact symbolic Jacobians.
//!
//! All items in this module require the `symbolic` feature flag.
//!
//! ## Variable Conventions
//!
//! The two solvers use **different** variable index conventions — do not mix them:
//!
//! | Solver                          | `Var(0)` | `Var(1..)` |
//! |---------------------------------|----------|------------|
//! | [`solve_ivp_symbolic`]          | time `t` | state `y[0], y[1], …` |
//! | [`quad_gauss_legendre_symbolic`]| `x`      | (unused)   |
//!
//! ## Example — BDF1 stiff ODE
//!
//! ```rust,ignore
//! use scirs2_integrate::eml::{solve_ivp_symbolic, SymbolicOdeError};
//! use scirs2_symbolic::eml::LoweredOp;
//! use scirs2_core::ndarray::arr1;
//! use std::sync::Arc;
//!
//! // x' = -1000·x  (stiff decay)
//! // Var(0) = t, Var(1) = x
//! let rhs = vec![Arc::new(LoweredOp::Mul(
//!     Box::new(LoweredOp::Const(-1000.0)),
//!     Box::new(LoweredOp::Var(1)),
//! ))];
//! let result = solve_ivp_symbolic(&rhs, [0.0, 0.001], arr1(&[1.0]).view(),
//!                                  1e-4, 1e-6, 1e-8, 200).unwrap();
//! ```

pub mod discover;
pub mod ode;
pub mod quad;

pub use discover::{discover_ode_from_trajectory, OdeDiscoveryConfig, OdeDiscoveryError};
pub use ode::{solve_ivp_symbolic, SymbolicOdeError, SymbolicOdeResult};
pub use quad::{quad_gauss_legendre_symbolic, SymbolicQuadError};
