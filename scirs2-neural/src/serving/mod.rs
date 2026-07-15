//! Model serving and deployment utilities for neural networks
//!
//! This module provides tools for packaging, deploying, and serving neural network models including:
//! - Model packaging with metadata and versioning
//! - C/C++ binding generation for native integration
//! - WebAssembly target compilation for web deployment
//! - Mobile deployment utilities for iOS/Android
//! - Runtime optimization and serving infrastructure
//! - Real, host-target code generation: [`ModelPackager`] can scaffold a throwaway
//!   Cargo project around a packaged model, compile it with `cargo build --release`,
//!   and copy out a genuine native executable or shared library. Targets other than
//!   the current host, and packaging formats that are not yet implemented, return an
//!   honest [`NeuralError`] rather than a fabricated artifact.

pub mod codegen;
pub mod packager;
#[cfg(test)]
pub mod tests;
pub mod types;

// Re-export all public types
pub use packager::ModelPackager;
pub use types::*;
