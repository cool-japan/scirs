//! Deadlock Detection and Prevention
//!
//! This module provides comprehensive deadlock detection, prevention, and recovery mechanisms
//! for distributed TPU synchronization including graph-based algorithms and resource management.
//!
//! This module has been refactored into focused submodules for better maintainability:
//! - `types`: Core types and configuration structures
//! - `algorithms`: Detection algorithms and optimization strategies
//! - `prevention`: Prevention strategies and policies
//! - `graph`: Dependency graph management and analysis
//! - `performance`: Performance monitoring and statistics
//! - `recovery`: Recovery coordination and execution
//! - `ml`: Machine learning components for prediction
//!
//! All original functionality is preserved through comprehensive re-exports.

// Re-export everything from the modular deadlock module
pub use deadlock::*;

// Include the modular deadlock implementation
pub mod deadlock;