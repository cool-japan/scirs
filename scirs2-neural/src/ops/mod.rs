//! Standalone neural network operation functions
//!
//! This module provides standalone mathematical operations used in neural networks,
//! operating directly on `ndarray` arrays. These complement the layer-based API
//! and are useful for building custom forward/backward passes.

pub mod activations;

pub use activations::{elu, gelu, leaky_relu, mish, relu, selu, sigmoid, softmax, swish, tanh};
