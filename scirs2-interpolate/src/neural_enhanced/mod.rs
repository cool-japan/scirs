//! Neural-network-enhanced interpolation.
//!
//! This module provides interpolation methods that combine traditional RBF
//! interpolation with a small trainable MLP to correct residuals.
//!
//! # Key types
//!
//! - [`residual_mlp_rbf::ResidualMlpRbf`] — fits RBF, trains MLP on residuals.
//! - [`tiny_mlp::TinyMlp`] — small MLP with analytic backpropagation.
//! - [`tiny_mlp::Activation`] — activation function enum (`Tanh`, `Relu`, `GeluApprox`).

pub mod residual_mlp_rbf;
pub mod tiny_mlp;

pub use residual_mlp_rbf::{ResidualMlpRbf, ResidualMlpRbfConfig};
pub use tiny_mlp::{Activation, TinyMlp};
