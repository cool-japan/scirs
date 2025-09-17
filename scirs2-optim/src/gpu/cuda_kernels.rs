//! Custom CUDA kernels for memory-intensive optimizers
//!
//! This module provides highly optimized CUDA kernel implementations for
//! memory-intensive optimizers like Adam and LAMB that can benefit from
//! custom GPU acceleration.
//!
//! The implementation has been refactored into a modular structure for better maintainability:
//! - Each component is separated into focused modules under `cuda_kernels/`
//! - All original functionality is preserved through comprehensive re-exports
//! - New features including Tensor Core acceleration, adaptive optimization,
//!   sophisticated memory management, and performance profiling are available
//! - Enhanced testing and documentation
//!
//! # Migration Note
//! All existing imports and usage patterns remain unchanged. The modular refactoring is
//! internal and does not affect the public API.

// Re-export all functionality from the modular implementation
pub use self::cuda_kernels::*;

// Declare the submodule
mod cuda_kernels;