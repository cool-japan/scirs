//! Bilevel optimization
//!
//! Provides bilevel programming methods where one optimization problem (the upper level)
//! is nested inside another (the lower level).
//!
//! # Mathematical Formulation
//!
//! ```text
//! min_{x,y}  F(x, y)              (upper level objective)
//! s.t.        G_i(x, y) <= 0      (upper level constraints)
//!             y solves:
//!               min_y  f(x, y)    (lower level objective)
//!               s.t.   g_j(x,y) <= 0  (lower level constraints)
//! ```
//!
//! # Methods
//!
//! - [`SingleLevelReduction`]: KKT-based single-level reformulation
//! - [`solve_bilevel_psoa`]: Penalty-based sequential optimization approach
//! - [`ReplacementAlgorithm`]: Replace lower level with optimal reaction mapping
//!
//! # Example
//!
//! ```no_run
//! use scirs2_optimize::bilevel::{
//!     BilevelProblem, BilevelSolverOptions, solve_bilevel_psoa, PsoaOptions,
//! };
//!
//! // Upper level: min_{x,y} (x - 1)^2 + (y - 1)^2
//! // Lower level: min_y (y - x)^2
//! let upper_obj = |x: &[f64], y: &[f64]| -> f64 {
//!     (x[0] - 1.0).powi(2) + (y[0] - 1.0).powi(2)
//! };
//! let lower_obj = |x: &[f64], y: &[f64]| -> f64 {
//!     (y[0] - x[0]).powi(2)
//! };
//!
//! let problem = BilevelProblem::new(
//!     upper_obj,
//!     lower_obj,
//!     vec![0.0],  // x0
//!     vec![0.0],  // y0
//! );
//!
//! let options = PsoaOptions::default();
//! let result = solve_bilevel_psoa(problem, options).expect("valid input");
//! println!("Upper x: {:?}, Lower y: {:?}", result.x_upper, result.y_lower);
//! ```

pub mod methods;

pub use methods::{
    solve_bilevel_psoa, solve_bilevel_replacement, solve_bilevel_single_level, BilevelProblem,
    BilevelResult, BilevelSolverOptions, PsoaOptions, ReplacementAlgorithm, SingleLevelReduction,
};
