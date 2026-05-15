//! MLE — Maximum Likelihood Estimation utilities.
//!
//! # Modules
//!
//! - [`mod@derive`] — symbolic MLE estimator factory via `cas::solve_system`.
//! - The `symbolic` sub-module re-exports the gradient-descent MLE from
//!   [`crate::mle_symbolic`] (only when the `symbolic` feature is active).

#[cfg(feature = "symbolic")]
pub mod derive;

#[cfg(feature = "symbolic")]
pub use derive::{derive, DeriveError, Estimator, FitError};
