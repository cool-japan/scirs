//! Machine Learning optimization module for SciRS2
//!
//! This module provides optimization algorithms specifically designed for machine learning,
//! including stochastic gradient descent variants, learning rate schedulers, and regularization techniques.
//!
//! # Features
//!
//! - Optimization algorithms: SGD, Adam, RMSprop, etc.
//! - Learning rate schedulers: ExponentialDecay, CosineAnnealing, etc.
//! - Regularization techniques: L1, L2, Dropout, etc.
//!
//! # Examples
//!
//! ```
//! use ndarray::{Array1, Array2};
//! use scirs2_optim::optimizers::{SGD, Optimizer};
//!
//! // Create a simple optimization problem
//! let params = Array1::zeros(5);
//! let gradients = Array1::from_vec(vec![0.1, 0.2, -0.3, 0.0, 0.5]);
//!
//! // Create an optimizer with learning rate 0.01
//! let mut optimizer = SGD::new(0.01);
//!
//! // Update parameters using the optimizer
//! let updated_params = optimizer.step(&params, &gradients);
//! // Parameters should be updated in the negative gradient direction
//! ```

#![warn(missing_docs)]

pub mod error;
pub mod gradient_processing;
pub mod memory_efficient;
pub mod metrics;
pub mod optimizer_composition;
pub mod optimizers;
pub mod parameter_groups;
pub mod regularizers;
pub mod schedulers;
pub mod utils;

// Re-exports for convenience
pub use gradient_processing::*;
#[cfg(feature = "metrics_integration")]
pub use metrics::*;
pub use optimizer_composition::*;
pub use optimizers::*;
pub use regularizers::*;
pub use schedulers::*;
