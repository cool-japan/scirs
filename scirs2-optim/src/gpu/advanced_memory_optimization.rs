//! Advanced GPU memory optimization for large-scale training
//!
//! This module provides advanced memory optimization techniques for large-scale
//! neural network training, including memory-efficient gradient accumulation,
//! activation checkpointing, and dynamic memory management.
//!
//! The implementation has been refactored into a modular structure for better maintainability:
//! - Each optimization technique is separated into focused modules under `advanced_memory_optimization/`
//! - All original functionality is preserved through comprehensive re-exports
//! - New convenience functions and improved APIs are available
//! - Enhanced testing and documentation
//!
//! # Migration Note
//! All existing imports and usage patterns remain unchanged. The modular refactoring is
//! internal and does not affect the public API.

// Re-export all functionality from the modular implementation
pub use self::advanced_memory_optimization::*;

// Declare the submodule
mod advanced_memory_optimization;