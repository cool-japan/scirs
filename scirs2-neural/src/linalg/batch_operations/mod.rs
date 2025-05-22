//! Batch matrix operations optimized for neural networks
//!
//! This module provides specialized batch matrix operations that are commonly
//! used in neural network computations, such as batch matrix multiplication,
//! batch normalization operations, convolution operations, attention mechanisms,
//! and RNN/LSTM operations.

mod attention;
mod convolution;
mod matmul;
mod normalization;
mod rnn;

// Re-export all public functions from submodules
pub use attention::*;
pub use convolution::*;
pub use matmul::*;
pub use normalization::*;
pub use rnn::*;
