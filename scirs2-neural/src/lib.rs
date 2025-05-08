//! Neural network building blocks module for SciRS2
//!
//! This module provides neural network building blocks for SciRS2, including:
//! - Layers (dense, convolutional, recurrent, etc.)
//! - Activation functions (ReLU, sigmoid, tanh, etc.)
//! - Loss functions (MSE, cross-entropy, etc.)
//! - Optimizers (SGD, Adam, etc.)
//! - Model architectures and training utilities
//! - Neural network specific linear algebra operations
//! - Model evaluation and testing
//! - Advanced training techniques

#![warn(missing_docs)]
#![recursion_limit = "524288"]

pub mod activations;
pub mod autograd;
pub mod callbacks;
// Temporarily disabled due to model config field mismatches
// pub mod config;
pub mod data;
pub mod error;
pub mod evaluation;
pub mod layers;
pub mod linalg;
pub mod losses;
pub mod models;
pub mod optimizers;
pub mod prelude;
pub mod serialization;
pub mod training;
pub mod transformer;
pub mod utils;

// Export specific items from each module to avoid name conflicts
// Use the prelude module for a convenient import

// Re-export the error type
pub use error::{Error, NeuralError, Result};
