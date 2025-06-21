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
/// Data augmentation module
pub mod augmentation;
/// C/C++ bindings module
pub mod bindings;
/// Model compression module
pub mod compression;
pub mod data;
/// Knowledge distillation module
pub mod distillation;
pub mod error;
pub mod evaluation;
/// GPU acceleration module (currently CPU fallback)
pub mod gpu;
/// Framework interoperability module
pub mod interop;
/// Interpretation module
pub mod interpretation;
pub mod layers;
pub mod linalg;
pub mod losses;
/// Memory-efficient operations module
pub mod memory_efficient;
/// Mobile deployment module
pub mod mobile;
/// Enhanced model evaluation module
pub mod model_evaluation;
pub mod models;
pub mod optimizers;
/// Performance optimization module
pub mod performance;
pub mod prelude;
/// Quantization module
pub mod quantization;
pub mod serialization;
/// Serving and deployment module
pub mod serving;
pub mod training;
/// Transfer learning module
pub mod transfer_learning;
pub mod transformer;
pub mod utils;
/// Visualization tools module
pub mod visualization;
/// WebAssembly module
pub mod wasm;

// Export specific items from each module to avoid name conflicts
// Use the prelude module for a convenient import

// Re-export the error type
pub use error::{Error, NeuralError, Result};
