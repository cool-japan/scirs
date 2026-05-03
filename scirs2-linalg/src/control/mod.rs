//! Control theory linear algebra tools.
//!
//! This module provides a collection of algorithms for control system analysis
//! and synthesis, organized into sub-modules:
//!
//! | Sub-module | Contents |
//! |------------|----------|
//! | [`lyapunov`] | Continuous and discrete Lyapunov equation solvers (Bartels-Stewart) |
//! | [`riccati`] | CARE and DARE solvers (Newton / doubling) |
//! | [`lqr`] | LQR controller synthesis, iLQR for nonlinear systems |
//! | [`stability`] | Controllability / observability analysis, Gramians, Hankel singular values |
//!
//! # Quick start
//!
//! ```rust
//! use scirs2_linalg::control::lqr::{LqrController, LqrMode};
//! use scirs2_core::ndarray::array;
//!
//! // Double integrator: x'' = u
//! let a = array![[0.0_f64, 1.0], [0.0, 0.0]];
//! let b = array![[0.0_f64], [1.0]];
//! let q = array![[1.0_f64, 0.0], [0.0, 1.0]]; // state cost
//! let r = array![[1.0_f64]];                   // input cost
//!
//! let ctrl = LqrController::new(
//!     &a.view(), &b.view(), &q.view(), &r.view(), LqrMode::Continuous,
//! ).expect("LQR synthesis failed");
//!
//! println!("Optimal gain K = {:?}", ctrl.gain);
//! println!("Riccati solution P = {:?}", ctrl.p);
//! ```
//!
//! # References
//! - Anderson, B. D. O. & Moore, J. B. (1990). *Optimal Control: Linear Quadratic Methods*.
//! - Skogestad, S. & Postlethwaite, I. (2005). *Multivariable Feedback Design*, 2nd ed.
//! - Lancaster, P. & Rodman, L. (1995). *Algebraic Riccati Equations*. Oxford.

pub mod lqr;
pub mod lyapunov;
pub mod riccati;
pub mod stability;

// ---------------------------------------------------------------------------
// Convenience re-exports
// ---------------------------------------------------------------------------

pub use lyapunov::{lyapunov_continuous, lyapunov_continuous_refine, lyapunov_discrete};

pub use riccati::{care_solve, dare_solve};

pub use lqr::{ilqr_solve, IlqrResult, LqrController, LqrMode};

pub use stability::{
    controllability_gramian, controllability_matrix, controllability_measure,
    hankel_singular_values, hautus_controllability_check, hautus_observability_check,
    is_controllable, is_observable, observability_gramian, observability_matrix,
    observability_measure,
};
